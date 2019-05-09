from __future__ import unicode_literals

import os
import sys
from importlib import import_module

from django.apps import apps
from django.conf import settings
from django.db.migrations.executor import \
    MigrationExecutor as DjangoMigrationExecutor
from django.db.migrations.graph import MigrationGraph
from django.db.migrations.recorder import MigrationRecorder
from django.db.migrations.state import ProjectState
from django.db.utils import DatabaseError
from django.utils import six
from django.utils.encoding import python_2_unicode_compatible


MIGRATIONS_MODULE_NAME = 'migrations'


class AmbiguityError(Exception):
    """
    Raised when more than one migration matches a name prefix.
    """
    pass


class BadMigrationError(Exception):
    """
    Raised when there's a bad migration (unreadable/bad format/etc.).
    """
    pass


class CircularDependencyError(Exception):
    """
    Raised when there's an impossible-to-resolve circular dependency.
    """
    pass


class InconsistentMigrationHistory(Exception):
    """
    Raised when an applied migration has some of its dependencies not applied.
    """
    pass


class InvalidBasesError(ValueError):
    """
    Raised when a model's base classes can't be resolved.
    """
    pass


class IrreversibleError(RuntimeError):
    """
    Raised when a irreversible migration is about to be reversed.
    """
    pass


@python_2_unicode_compatible
class NodeNotFoundError(LookupError):
    """
    Raised when an attempt on a node is made that is not available in the graph.
    """

    def __init__(self, message, node, origin=None):
        self.message = message
        self.origin = origin
        self.node = node

    def __str__(self):
        return self.message

    def __repr__(self):
        return "NodeNotFoundError(%r)" % (self.node, )


class MigrationSchemaMissing(DatabaseError):
    pass


class InvalidMigrationPlan(ValueError):
    pass


class MigrationLoader(object):
    """
    Loads migration files from disk, and their status from the database.

    Migration files are expected to live in the "migrations" directory of
    an app. Their names are entirely unimportant from a code perspective,
    but will probably follow the 1234_name.py convention.

    On initialization, this class will scan those directories, and open and
    read the python files, looking for a class called Migration, which should
    inherit from django.db.migrations.Migration. See
    django.db.migrations.migration for what that looks like.

    Some migrations will be marked as "replacing" another set of migrations.
    These are loaded into a separate set of migrations away from the main ones.
    If all the migrations they replace are either unapplied or missing from
    disk, then they are injected into the main set, replacing the named migrations.
    Any dependency pointers to the replaced migrations are re-pointed to the
    new migration.

    This does mean that this class MUST also talk to the database as well as
    to disk, but this is probably fine. We're already not just operating
    in memory.
    """

    def __init__(self, connection, load=True, ignore_no_migrations=False):
        self.connection = connection
        self.disk_migrations = None
        self.applied_migrations = None
        self.ignore_no_migrations = ignore_no_migrations
        if load:
            self.build_graph()

    @classmethod
    def migrations_module(cls, app_label):
        if app_label in settings.MIGRATION_MODULES:
            return settings.MIGRATION_MODULES[app_label]
        else:
            app_package_name = apps.get_app_config(app_label).name
            return '%s.%s' % (app_package_name, MIGRATIONS_MODULE_NAME)

    def load_disk(self):
        """
        Loads the migrations from all INSTALLED_APPS from disk.
        """
        self.disk_migrations = {}
        self.unmigrated_apps = set()
        self.migrated_apps = set()
        for app_config in apps.get_app_configs():
            # Get the migrations module directory
            module_name = self.migrations_module(app_config.label)
            if module_name is None:
                self.unmigrated_apps.add(app_config.label)
                continue
            was_loaded = module_name in sys.modules
            try:
                module = import_module(module_name)
            except ImportError as e:
                # I hate doing this, but I don't want to squash other import errors.
                # Might be better to try a directory check directly.
                if "No module named" in str(e) and MIGRATIONS_MODULE_NAME in str(e):
                    self.unmigrated_apps.add(app_config.label)
                    continue
                raise
            else:
                # PY3 will happily import empty dirs as namespaces.
                if not hasattr(module, '__file__'):
                    self.unmigrated_apps.add(app_config.label)
                    continue
                # Module is not a package (e.g. migrations.py).
                if not hasattr(module, '__path__'):
                    self.unmigrated_apps.add(app_config.label)
                    continue
                # Force a reload if it's already loaded (tests need this)
                if was_loaded:
                    six.moves.reload_module(module)
            self.migrated_apps.add(app_config.label)
            directory = os.path.dirname(module.__file__)
            # Scan for .py files
            migration_names = set()
            for name in os.listdir(directory):
                if name.endswith(".py"):
                    import_name = name.rsplit(".", 1)[0]
                    if import_name[0] not in "_.~":
                        migration_names.add(import_name)
            # Load them
            for migration_name in migration_names:
                migration_module = import_module("%s.%s" % (module_name, migration_name))
                if not hasattr(migration_module, "Migration"):
                    raise BadMigrationError(
                        "Migration %s in app %s has no Migration class" % (migration_name, app_config.label)
                    )
                self.disk_migrations[app_config.label, migration_name] = migration_module.Migration(
                    migration_name,
                    app_config.label,
                )

    def get_migration(self, app_label, name_prefix):
        "Gets the migration exactly named, or raises `graph.NodeNotFoundError`"
        return self.graph.nodes[app_label, name_prefix]

    def get_migration_by_prefix(self, app_label, name_prefix):
        "Returns the migration(s) which match the given app label and name _prefix_"
        # Do the search
        results = []
        for l, n in self.disk_migrations:
            if l == app_label and n.startswith(name_prefix):
                results.append((l, n))
        if len(results) > 1:
            raise AmbiguityError(
                "There is more than one migration for '%s' with the prefix '%s'" % (app_label, name_prefix)
            )
        elif len(results) == 0:
            raise KeyError("There no migrations for '%s' with the prefix '%s'" % (app_label, name_prefix))
        else:
            return self.disk_migrations[results[0]]

    def check_key(self, key, current_app):
        if (key[1] != "__first__" and key[1] != "__latest__") or key in self.graph:
            return key
        # Special-case __first__, which means "the first migration" for
        # migrated apps, and is ignored for unmigrated apps. It allows
        # makemigrations to declare dependencies on apps before they even have
        # migrations.
        if key[0] == current_app:
            # Ignore __first__ references to the same app (#22325)
            return
        if key[0] in self.unmigrated_apps:
            # This app isn't migrated, but something depends on it.
            # The models will get auto-added into the state, though
            # so we're fine.
            return
        if key[0] in self.migrated_apps:
            try:
                if key[1] == "__first__":
                    return list(self.graph.root_nodes(key[0]))[0]
                else:  # "__latest__"
                    return list(self.graph.leaf_nodes(key[0]))[0]
            except IndexError:
                if self.ignore_no_migrations:
                    return None
                else:
                    raise ValueError("Dependency on app with no migrations: %s" % key[0])
        raise ValueError("Dependency on unknown app: %s" % key[0])

    def add_internal_dependencies(self, key, migration):
        """
        Internal dependencies need to be added first to ensure `__first__`
        dependencies find the correct root node.
        """
        for parent in migration.dependencies:
            if parent[0] != key[0] or parent[1] == '__first__':
                # Ignore __first__ references to the same app (#22325).
                continue
            self.graph.add_dependency(migration, key, parent, skip_validation=True)

    def add_external_dependencies(self, key, migration):
        for parent in migration.dependencies:
            # Skip internal dependencies
            if key[0] == parent[0]:
                continue
            parent = self.check_key(parent, key[0])
            if parent is not None:
                self.graph.add_dependency(migration, key, parent, skip_validation=True)
        for child in migration.run_before:
            child = self.check_key(child, key[0])
            if child is not None:
                self.graph.add_dependency(migration, child, key, skip_validation=True)

    def build_graph(self):
        """
        Builds a migration dependency graph using both the disk and database.
        You'll need to rebuild the graph if you apply migrations. This isn't
        usually a problem as generally migration stuff runs in a one-shot process.
        """
        # Load disk data
        self.load_disk()
        # Load database data
        if self.connection is None:
            self.applied_migrations = set()
        else:
            recorder = MigrationRecorder(self.connection)
            self.applied_migrations = recorder.applied_migrations()
        # To start, populate the migration graph with nodes for ALL migrations
        # and their dependencies. Also make note of replacing migrations at this step.
        self.graph = MigrationGraph()
        self.replacements = {}
        for key, migration in self.disk_migrations.items():
            self.graph.add_node(key, migration)
            # Internal (aka same-app) dependencies.
            self.add_internal_dependencies(key, migration)
            # Replacing migrations.
            if migration.replaces:
                self.replacements[key] = migration
        # Add external dependencies now that the internal ones have been resolved.
        for key, migration in self.disk_migrations.items():
            self.add_external_dependencies(key, migration)
        # Carry out replacements where possible.
        for key, migration in self.replacements.items():
            # Get applied status of each of this migration's replacement targets.
            applied_statuses = [(target in self.applied_migrations) for target in migration.replaces]
            # Ensure the replacing migration is only marked as applied if all of
            # its replacement targets are.
            if all(applied_statuses):
                self.applied_migrations.add(key)
            else:
                self.applied_migrations.discard(key)
            # A replacing migration can be used if either all or none of its
            # replacement targets have been applied.
            if all(applied_statuses) or (not any(applied_statuses)):
                self.graph.remove_replaced_nodes(key, migration.replaces)
            else:
                # This replacing migration cannot be used because it is partially applied.
                # Remove it from the graph and remap dependencies to it (#25945).
                self.graph.remove_replacement_node(key, migration.replaces)
        # Ensure the graph is consistent.
        try:
            self.graph.validate_consistency()
        except NodeNotFoundError as exc:
            # Check if the missing node could have been replaced by any squash
            # migration but wasn't because the squash migration was partially
            # applied before. In that case raise a more understandable exception
            # (#23556).
            # Get reverse replacements.
            reverse_replacements = {}
            for key, migration in self.replacements.items():
                for replaced in migration.replaces:
                    reverse_replacements.setdefault(replaced, set()).add(key)
            # Try to reraise exception with more detail.
            if exc.node in reverse_replacements:
                candidates = reverse_replacements.get(exc.node, set())
                is_replaced = any(candidate in self.graph.nodes for candidate in candidates)
                if not is_replaced:
                    tries = ', '.join('%s.%s' % c for c in candidates)
                    exc_value = NodeNotFoundError(
                        "Migration {0} depends on nonexistent node ('{1}', '{2}'). "
                        "Django tried to replace migration {1}.{2} with any of [{3}] "
                        "but wasn't able to because some of the replaced migrations "
                        "are already applied.".format(
                            exc.origin, exc.node[0], exc.node[1], tries
                        ),
                        exc.node
                    )
                    exc_value.__cause__ = exc
                    if not hasattr(exc, '__traceback__'):
                        exc.__traceback__ = sys.exc_info()[2]
                    six.reraise(NodeNotFoundError, exc_value, sys.exc_info()[2])
            raise exc

    def check_consistent_history(self, connection):
        """
        Raise InconsistentMigrationHistory if any applied migrations have
        unapplied dependencies.
        """
        recorder = MigrationRecorder(connection)
        applied = recorder.applied_migrations()
        for migration in applied:
            # If the migration is unknown, skip it.
            if migration not in self.graph.nodes:
                continue
            for parent in self.graph.node_map[migration].parents:
                if parent not in applied:
                    # Skip unapplied squashed migrations that have all of their
                    # `replaces` applied.
                    if parent in self.replacements:
                        if all(m in applied for m in self.replacements[parent].replaces):
                            continue
                    raise InconsistentMigrationHistory(
                        "Migration {}.{} is applied before its dependency "
                        "{}.{} on database '{}'.".format(
                            migration[0], migration[1], parent[0], parent[1],
                            connection.alias,
                        )
                    )

    def detect_conflicts(self):
        """
        Looks through the loaded graph and detects any conflicts - apps
        with more than one leaf migration. Returns a dict of the app labels
        that conflict with the migration names that conflict.
        """
        seen_apps = {}
        conflicting_apps = set()
        for app_label, migration_name in self.graph.leaf_nodes():
            if app_label in seen_apps:
                conflicting_apps.add(app_label)
            seen_apps.setdefault(app_label, set()).add(migration_name)
        return {app_label: seen_apps[app_label] for app_label in conflicting_apps}

    def project_state(self, nodes=None, at_end=True):
        """
        Returns a ProjectState object representing the most recent state
        that the migrations we loaded represent.

        See graph.make_state for the meaning of "nodes" and "at_end"
        """
        return self.graph.make_state(nodes=nodes, at_end=at_end, real_apps=list(self.unmigrated_apps))


class MigrationExecutor(DjangoMigrationExecutor):
    """
    Changed the main method, migrate(), to a backport of the
    method from 1.10 which runs forward migrations a lot faster.
    See Django tickets 24743, 24745, 24100, 26647, 27044, 27100
    and related commits.
    """

    def migrate(self, targets, plan=None, fake=False, fake_initial=False):
        """
        Migrates the database up to the given targets.

        Django first needs to create all project states before a migration is
        (un)applied and in a second step run all the database operations.
        """
        if plan is None:
            plan = self.migration_plan(targets)
        # Create the forwards plan Django would follow on an empty database
        full_plan = self.migration_plan(self.loader.graph.leaf_nodes(), clean_start=True)

        all_forwards = all(not backwards for mig, backwards in plan)
        all_backwards = all(backwards for mig, backwards in plan)

        if not plan:
            pass  # Nothing to do for an empty plan
        elif all_forwards == all_backwards:
            # This should only happen if there's a mixed plan
            raise InvalidMigrationPlan(
                "Migration plans with both forwards and backwards migrations "
                "are not supported. Please split your migration process into "
                "separate plans of only forwards OR backwards migrations.",
                plan
            )
        elif all_forwards:
            self._migrate_all_forwards(plan, full_plan, fake=fake, fake_initial=fake_initial)
        else:
            # No need to check for `elif all_backwards` here, as that condition
            # would always evaluate to true.
            self._migrate_all_backwards(plan, full_plan, fake=fake)

        self.check_replacements()

    def _migrate_all_forwards(self, plan, full_plan, fake, fake_initial):
        """
        Take a list of 2-tuples of the form (migration instance, False) and
        apply them in the order they occur in the full_plan.
        """
        state = self._create_project_state(with_applied_migrations=True)
        migrations_to_run = {m[0] for m in plan}
        for migration, _ in full_plan:
            if not migrations_to_run:
                # We remove every migration that we applied from this set so
                # that we can bail out once the last migration has been applied
                # and don't always run until the very end of the migration
                # process.
                break
            if migration in migrations_to_run:
                if 'apps' not in state.__dict__:
                    if self.progress_callback:
                        self.progress_callback("render_start")
                    state.apps  # Render all -- performance critical
                    if self.progress_callback:
                        self.progress_callback("render_success")
                state = self.apply_migration(state, migration, fake=fake, fake_initial=fake_initial)
                migrations_to_run.remove(migration)

        return state

    def _create_project_state(self, with_applied_migrations=False):
        """
        Create a project state including all the applications without
        migrations and applied migrations if with_applied_migrations=True.
        """
        state = ProjectState(real_apps=list(self.loader.unmigrated_apps))
        if with_applied_migrations:
            # Create the forwards plan Django would follow on an empty database
            full_plan = self.migration_plan(self.loader.graph.leaf_nodes(), clean_start=True)
            applied_migrations = {
                self.loader.graph.nodes[key] for key in self.loader.applied_migrations
                if key in self.loader.graph.nodes
            }
            for migration, _ in full_plan:
                if migration in applied_migrations:
                    migration.mutate_state(state, preserve=False)
        return state

    def _migrate_all_backwards(self, plan, full_plan, fake):
        """
        Take a list of 2-tuples of the form (migration instance, True) and
        unapply them in reverse order they occur in the full_plan.

        Since unapplying a migration requires the project state prior to that
        migration, Django will compute the migration states before each of them
        in a first run over the plan and then unapply them in a second run over
        the plan.
        """
        migrations_to_run = {m[0] for m in plan}
        # Holds all migration states prior to the migrations being unapplied
        states = {}
        state = ProjectState(real_apps=list(self.loader.unmigrated_apps))
        if self.progress_callback:
            self.progress_callback("render_start")
        for migration, _ in full_plan:
            if not migrations_to_run:
                # We remove every migration that we applied from this set so
                # that we can bail out once the last migration has been applied
                # and don't always run until the very end of the migration
                # process.
                break
            if migration in migrations_to_run:
                if 'apps' not in state.__dict__:
                    state.apps  # Render all -- performance critical
                # The state before this migration
                states[migration] = state
                # The old state keeps as-is, we continue with the new state
                state = migration.mutate_state(state, preserve=True)
                migrations_to_run.remove(migration)
            else:
                migration.mutate_state(state, preserve=False)
        if self.progress_callback:
            self.progress_callback("render_success")

        for migration, _ in plan:
            self.unapply_migration(states[migration], migration, fake=fake)

    def check_replacements(self):
        """
        Mark replacement migrations applied if their replaced set all are.

        We do this unconditionally on every migrate, rather than just when
        migrations are applied or unapplied, so as to correctly handle the case
        when a new squash migration is pushed to a deployment that already had
        all its replaced migrations applied. In this case no new migration will
        be applied, but we still want to correctly maintain the applied state
        of the squash migration.
        """
        applied = self.recorder.applied_migrations()
        for key, migration in self.loader.replacements.items():
            all_applied = all(m in applied for m in migration.replaces)
            if all_applied and key not in applied:
                self.recorder.record_applied(*key)


def monkeypatch(module):
    module.MigrationExecutor = MigrationExecutor

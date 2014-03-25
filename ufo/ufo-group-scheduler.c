/*
 * Copyright (C) 2011-2013 Karlsruhe Institute of Technology
 *
 * This file is part of Ufo.
 *
 * This library is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "config.h"

#ifdef HAVE_PYTHON
#include <Python.h>
#endif

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <gio/gio.h>
#include <stdio.h>
#include <string.h>

#include <ufo/ufo-buffer.h>
#include <ufo/ufo-config.h>
#include <ufo/ufo-configurable.h>
#include <ufo/ufo-resources.h>
#include <ufo/ufo-group-scheduler.h>
#include <ufo/ufo-task-node.h>
#include <ufo/ufo-task-iface.h>
#include <ufo/ufo-two-way-queue.h>
#include "compat.h"

/**
 * SECTION:ufo-scheduler
 * @Short_description: Schedule the execution of a graph of nodes
 * @Title: UfoGroupScheduler
 *
 * A scheduler object uses a graphs information to schedule the contained nodes
 * on CPU and GPU hardware.
 */

static void ufo_group_scheduler_initable_iface_init (GInitableIface *iface);

G_DEFINE_TYPE_WITH_CODE (UfoGroupScheduler, ufo_group_scheduler, G_TYPE_OBJECT,
                         G_IMPLEMENT_INTERFACE (UFO_TYPE_CONFIGURABLE, NULL)
                         G_IMPLEMENT_INTERFACE (G_TYPE_INITABLE,
                                                ufo_group_scheduler_initable_iface_init))

#define UFO_GROUP_SCHEDULER_GET_PRIVATE(obj) (G_TYPE_INSTANCE_GET_PRIVATE((obj), UFO_TYPE_GROUP_SCHEDULER, UfoGroupSchedulerPrivate))


struct _UfoGroupSchedulerPrivate {
    GError          *construct_error;
    UfoConfig       *config;
    UfoResources    *resources;
    UfoArchGraph    *arch_graph;
    gdouble          time;
};

enum {
    PROP_0,
    PROP_TIME,
    N_PROPERTIES,

    /* Here come the overriden properties that we don't install ourselves. */
    PROP_CONFIG,
};

static GParamSpec *properties[N_PROPERTIES] = { NULL, };

/**
 * UfoGroupSchedulerError:
 * @UFO_GROUP_SCHEDULER_ERROR_SETUP: Could not start scheduler due to error
 */
GQuark
ufo_group_scheduler_error_quark (void)
{
    return g_quark_from_static_string ("ufo-scheduler-error-quark");
}


/**
 * ufo_group_scheduler_new:
 * @config: A #UfoConfig or %NULL
 *
 * Creates a new #UfoGroupScheduler.
 *
 * Return value: A new #UfoGroupScheduler
 */
UfoGroupScheduler *
ufo_group_scheduler_new (UfoConfig *config)
{
    return UFO_GROUP_SCHEDULER (g_object_new (UFO_TYPE_GROUP_SCHEDULER,
                                              "config", config,
                                              NULL));
}

/**
 * ufo_group_scheduler_get_context:
 * @scheduler: A #UfoGroupScheduler
 *
 * Get the associated OpenCL context of @scheduler.
 *
 * Return value: (transfer full): An cl_context structure or %NULL on error.
 */
gpointer
ufo_group_scheduler_get_context (UfoGroupScheduler *scheduler)
{
    g_return_val_if_fail (UFO_IS_GROUP_SCHEDULER (scheduler), NULL);
    return ufo_resources_get_context (scheduler->priv->resources);
}

void
ufo_group_scheduler_set_arch_graph (UfoGroupScheduler *scheduler,
                              UfoArchGraph *graph)
{
    g_return_if_fail (UFO_IS_GROUP_SCHEDULER (scheduler));

    if (scheduler->priv->arch_graph != NULL)
        g_object_unref (scheduler->priv->arch_graph);

    scheduler->priv->arch_graph = g_object_ref (graph);
}

/**
 * ufo_group_scheduler_get_resources:
 * @scheduler: A #UfoGroupScheduler
 *
 * Get a reference on the #UfoResources object of this scheduler.
 *
 * Return value: (transfer none): Associated #UfoResources object.
 */
UfoResources *
ufo_group_scheduler_get_resources (UfoGroupScheduler *scheduler)
{
    g_return_val_if_fail (UFO_IS_GROUP_SCHEDULER (scheduler), NULL);
    return scheduler->priv->resources;
}

typedef struct {
    GList *parents;
    GList *tasks;
    gboolean is_leaf;
    gpointer context;
    UfoTwoWayQueue *queue;
    enum {
        TASK_GROUP_ROUND_ROBIN,
        TASK_GROUP_SHARED,
        TASK_GROUP_RANDOM
    } mode;
} TaskGroup;


static void
expand_group_graph (UfoGraph *graph, UfoArchGraph *arch)
{
    GList *nodes;
    GList *it;
    GList *gpu_nodes;
    guint n_gpus;

    gpu_nodes = ufo_arch_graph_get_gpu_nodes (arch);
    n_gpus = g_list_length (gpu_nodes);

    nodes = ufo_graph_get_nodes (graph);

    g_list_for (nodes, it) {
        TaskGroup *group;
        UfoTaskNode *node;

        group = ufo_node_get_label (UFO_NODE (it->data));
        node = UFO_TASK_NODE (group->tasks->data);

        if (ufo_task_uses_gpu (UFO_TASK (node))) {
            ufo_task_node_set_proc_node (node, g_list_first (gpu_nodes)->data);

            for (guint i = 1; i < n_gpus; i++) {
                UfoTaskNode *copy;
                GError *error = NULL;

                copy = UFO_TASK_NODE (ufo_node_copy (UFO_NODE (node), &error));
                ufo_task_node_set_proc_node (copy, g_list_nth_data (gpu_nodes, i));

                if (error != NULL)
                    g_print ("error copying: %s\n", error->message);

                group->tasks = g_list_append (group->tasks, copy);
            }
        }
    }

    g_list_free (gpu_nodes);
    g_list_free (nodes);
}

static UfoGraph *
build_group_graph (UfoGroupSchedulerPrivate *priv, UfoTaskGraph *graph, UfoArchGraph *arch)
{
    UfoGraph *result;
    GList *nodes;
    GList *it;
    GHashTable *tasks_to_groups;

    result = ufo_graph_new ();
    tasks_to_groups = g_hash_table_new (g_direct_hash, g_direct_equal);
    nodes = ufo_graph_get_nodes (UFO_GRAPH (graph));

    /* Create a group with a single member for each node */
    g_list_for (nodes, it) {
        TaskGroup *group;
        UfoNode *node;

        group = g_new0 (TaskGroup, 1);
        group->context = ufo_resources_get_context (priv->resources);
        group->parents = NULL;
        group->tasks = g_list_append (NULL, it->data);
        group->queue = ufo_two_way_queue_new (NULL);
        group->mode = TASK_GROUP_ROUND_ROBIN;
        group->is_leaf = ufo_graph_get_num_successors (UFO_GRAPH (graph), UFO_NODE (it->data)) == 0;

        node = ufo_node_new (group);
        g_hash_table_insert (tasks_to_groups, it->data, node);
    }

    /* Link groups */
    g_list_for (nodes, it) {
        GList *predecessors;
        GList *jt;
        TaskGroup *group;
        UfoNode *group_node;

        group_node = g_hash_table_lookup (tasks_to_groups, it->data);
        group = ufo_node_get_label (group_node);
        predecessors = ufo_graph_get_predecessors (UFO_GRAPH (graph), group->tasks->data);

        /* Connect predecessors to current node */
        g_list_for (predecessors, jt) {
            UfoNode *parent_node;
            UfoGroup *parent_group;

            parent_node = g_hash_table_lookup (tasks_to_groups, jt->data);
            parent_group = ufo_node_get_label (parent_node);
            group->parents = g_list_append (group->parents, parent_group);

            /* FIXME: use correct input as label */
            ufo_graph_connect_nodes (result, parent_node, group_node, NULL);
        }

        g_list_free (predecessors);
    }

    g_list_free (nodes);
    g_hash_table_destroy (tasks_to_groups);

    expand_group_graph (result, arch);
    return result;
}

static UfoBuffer *poisonpill = (UfoBuffer *) 0x1;

static GError *
run_group (TaskGroup *group)
{
    GList *it;
    guint n_inputs;
    guint i;
    UfoBuffer **inputs;
    UfoBuffer *output;
    UfoRequisition requisition;
    UfoTask *task;
    gboolean finished = FALSE;
    GList *current = NULL;  /* current task */

    /* We should use get_structure to assert constraints ... */
    n_inputs = g_list_length (group->parents);
    inputs = g_new0 (UfoBuffer *, n_inputs);

    while (!finished) {
        /* Choose next task of the group */
        current = g_list_next (current);

        if (current == NULL)
            current = g_list_first (group->tasks);

        task = UFO_TASK (current->data);

        i = 0;

        /* Fetch data from parent groups */
        g_list_for (group->parents, it) {
            UfoTwoWayQueue *queue;

            queue = ((TaskGroup *) it->data)->queue;
            inputs[i] = ufo_two_way_queue_consumer_pop (queue);

            if (inputs[i] == poisonpill)
                finished = TRUE;

            i++;
        }

        if (finished)
            break;

        /* Ask current task about size requirements */
        ufo_task_get_requisition (task, inputs, &requisition);

        if (!group->is_leaf) {
            /* Insert buffers as longs as capacity is not filled */
            if (ufo_two_way_queue_get_capacity (group->queue) < 2) {
                UfoBuffer *buffer;

                buffer = ufo_buffer_new (&requisition, group->context);
                ufo_two_way_queue_insert (group->queue, buffer);
            }

            output = ufo_two_way_queue_producer_pop (group->queue);
        }

        /* Generate/process the data. Because the functions return active state,
         * we negate it for the finished flag. */

        if (n_inputs == 0) {
            finished = !ufo_task_generate (task, output, &requisition);
        }
        else {
            finished = !ufo_task_process (task, inputs, output, &requisition);
        }

        if (finished)
            break;

        i = 0;

        g_list_for (group->parents, it) {
            UfoTwoWayQueue *queue;

            queue = ((TaskGroup *) it->data)->queue;
            ufo_two_way_queue_consumer_push (queue, inputs[i++]);
        }

        ufo_two_way_queue_producer_push (group->queue, output);
    }

    if (!group->is_leaf)
        ufo_two_way_queue_producer_push (group->queue, poisonpill);

    return NULL;
}

static void
join_threads (GList *threads)
{
    GList *it;

    g_list_for (threads, it) {
        g_thread_join (it->data);
    }
}

void
ufo_group_scheduler_run (UfoGroupScheduler *scheduler,
                         UfoTaskGraph *task_graph,
                         GError **error)
{
    UfoGroupSchedulerPrivate *priv;
    UfoArchGraph *arch_graph;
    UfoGraph *group_graph;
    GTimer *timer;
    GList *threads;
    GList *groups;
    GList *it;

    g_return_if_fail (UFO_IS_GROUP_SCHEDULER (scheduler));

    priv = scheduler->priv;

    if (priv->construct_error != NULL) {
        if (error)
            *error = g_error_copy (priv->construct_error);
        return;
    }

    if (priv->arch_graph != NULL) {
        arch_graph = priv->arch_graph;
    }
    else {
        arch_graph = UFO_ARCH_GRAPH (ufo_arch_graph_new (priv->resources, NULL));
    }

    timer = g_timer_new ();

    group_graph = build_group_graph (priv, task_graph, arch_graph);
    groups = ufo_graph_get_nodes (group_graph);

    g_list_for (groups, it) {
        GThread *thread;
        GList *jt;
        TaskGroup *group;

        group = ufo_node_get_label (UFO_NODE (it->data));

        /* Setup each task in a group */
        g_list_for (group->tasks, jt) {
            ufo_task_setup (UFO_TASK (jt->data), priv->resources, error);

            if (error && *error)
                goto cleanup_run;
        }

        /* Run this group */
        thread = g_thread_create ((GThreadFunc) run_group, group, TRUE, error);
        threads = g_list_append (threads, thread);
    }

#ifdef HAVE_PYTHON
    if (Py_IsInitialized ()) {
        Py_BEGIN_ALLOW_THREADS

        join_threads (threads);

        Py_END_ALLOW_THREADS
    }
    else {
        join_threads (threads);
    }
#else
    join_threads (threads);
#endif

cleanup_run:
    g_list_free (groups);

    priv->time = g_timer_elapsed (timer, NULL);
    g_message ("Processing finished after %3.5fs", priv->time);
    g_timer_destroy (timer);

    g_object_unref (group_graph);

    if (priv->arch_graph == NULL)
        g_object_unref (arch_graph);
}

static void
ufo_group_scheduler_set_property (GObject      *object,
                                  guint         property_id,
                                  const GValue *value,
                                  GParamSpec   *pspec)
{
    UfoGroupSchedulerPrivate *priv = UFO_GROUP_SCHEDULER_GET_PRIVATE (object);

    switch (property_id) {
        case PROP_CONFIG:
            {
                GObject *vobject = g_value_get_object (value);

                if (vobject != NULL) {
                    if (priv->config != NULL)
                        g_object_unref (priv->config);

                    priv->config = UFO_CONFIG (vobject);
                    g_object_ref (priv->config);
                }
            }
            break;

        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
            break;
    }
}

static void
ufo_group_scheduler_get_property (GObject      *object,
                                  guint         property_id,
                                  GValue       *value,
                                  GParamSpec   *pspec)
{
    UfoGroupSchedulerPrivate *priv = UFO_GROUP_SCHEDULER_GET_PRIVATE (object);

    switch (property_id) {
        case PROP_TIME:
            g_value_set_double (value, priv->time);
            break;

        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
            break;
    }
}

static void
ufo_group_scheduler_constructed (GObject *object)
{
    UfoGroupSchedulerPrivate *priv;

    priv = UFO_GROUP_SCHEDULER_GET_PRIVATE (object);
    priv->resources = ufo_resources_new (priv->config,
                                         &priv->construct_error);
}

static void
ufo_group_scheduler_dispose (GObject *object)
{
    UfoGroupSchedulerPrivate *priv;

    priv = UFO_GROUP_SCHEDULER_GET_PRIVATE (object);

    if (priv->config != NULL) {
        g_object_unref (priv->config);
        priv->config = NULL;
    }

    if (priv->resources != NULL) {
        g_object_unref (priv->resources);
        priv->resources = NULL;
    }

    if (priv->arch_graph != NULL) {
        g_object_unref (priv->arch_graph);
        priv->arch_graph = NULL;
    }

    G_OBJECT_CLASS (ufo_group_scheduler_parent_class)->dispose (object);
}

static void
ufo_group_scheduler_finalize (GObject *object)
{
    UfoGroupSchedulerPrivate *priv;

    priv = UFO_GROUP_SCHEDULER_GET_PRIVATE (object);

    g_clear_error (&priv->construct_error);

    G_OBJECT_CLASS (ufo_group_scheduler_parent_class)->finalize (object);
}

static gboolean
ufo_group_scheduler_initable_init (GInitable *initable,
                                   GCancellable *cancellable,
                                   GError **error)
{
    UfoGroupScheduler *scheduler;
    UfoGroupSchedulerPrivate *priv;

    g_return_val_if_fail (UFO_IS_GROUP_SCHEDULER (initable), FALSE);

    if (cancellable != NULL) {
        g_set_error_literal (error, G_IO_ERROR, G_IO_ERROR_NOT_SUPPORTED,
                             "Cancellable initialization not supported");
        return FALSE;
    }

    scheduler = UFO_GROUP_SCHEDULER (initable);
    priv = scheduler->priv;

    if (priv->construct_error != NULL) {
        if (error)
            *error = g_error_copy (priv->construct_error);

        return FALSE;
    }

    return TRUE;
}

static void
ufo_group_scheduler_initable_iface_init (GInitableIface *iface)
{
    iface->init = ufo_group_scheduler_initable_init;
}

static void
ufo_group_scheduler_class_init (UfoGroupSchedulerClass *klass)
{
    GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
    gobject_class->constructed  = ufo_group_scheduler_constructed;
    gobject_class->set_property = ufo_group_scheduler_set_property;
    gobject_class->get_property = ufo_group_scheduler_get_property;
    gobject_class->dispose      = ufo_group_scheduler_dispose;
    gobject_class->finalize     = ufo_group_scheduler_finalize;

    properties[PROP_TIME] =
        g_param_spec_double ("time",
                             "Finished execution time",
                             "Finished execution time in seconds",
                              0.0, G_MAXDOUBLE, 0.0,
                              G_PARAM_READABLE);

    for (guint i = PROP_0 + 1; i < N_PROPERTIES; i++)
        g_object_class_install_property (gobject_class, i, properties[i]);

    g_object_class_override_property (gobject_class, PROP_CONFIG, "config");

    g_type_class_add_private (klass, sizeof (UfoGroupSchedulerPrivate));
}

static void
ufo_group_scheduler_init (UfoGroupScheduler *scheduler)
{
    UfoGroupSchedulerPrivate *priv;

    scheduler->priv = priv = UFO_GROUP_SCHEDULER_GET_PRIVATE (scheduler);
    priv->config = NULL;
    priv->resources = NULL;
    priv->arch_graph = NULL;
    priv->construct_error = NULL;
    priv->time = 0.0;
}

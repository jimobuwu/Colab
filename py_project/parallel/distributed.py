import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.app.flags.DEFINE_string("ps_hosts", "localhost:2222", "ps hosts")
tf.app.flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224", "worker hosts")
tf.app.flags.DEFINE_string("job_name", "worker", "'ps' or'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("num_workers", 2, "Number of workers")
tf.app.flags.DEFINE_boolean("is_sync", False, "using synchronous training or not")

FLAGS = tf.app.flags.FLAGS


def model(images):
    net = tf.layers.dense(images, 500, activation=tf.nn.relu)
    net = tf.layers.dense(net, 500, activation = tf.nn.relu)
    net = tf.layers.dense(net, 10)
    return net


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Clusterspec
    cluster = tf.train.ClusterSpec({
        "ps": ps_hosts,
        "worker": worker_hosts
    })

    # Server
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        # building a Graph for replicas
        # ps_strategy默认使用round_robin,参数按顺序依次循环部署到ps
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index,
                       cluster=cluster)):
            mnist = input_data.read_data_sets("../data/mnist_data", one_hot=True)
            images = tf.placeholder(tf.float32, [None, 784])
            labels = tf.placeholder(tf.int32, [None, 10])

            logits = model(images)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
            hooks = [tf.train.StopAtStepHook(last_step=2000)]

            global_step = tf.train.get_or_create_global_step()
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)

            # 同步梯度更新
            if FLAGS.is_sync:
                # wrapper
                optimizer = tf.train.SyncReplicasOptimizer(optimizer,
                                                           replicas_to_aggregate=FLAGS.num_workers,
                                                           total_num_replicas=FLAGS.num_workers)
                hooks.append(optimizer.make_session_run_hook(is_chief=(FLAGS.task_index == 0)))

            # aggregation_method,合并梯度的方法
            train_op = optimizer.minimize(loss,
                                          global_step=global_step,
                                          aggregation_method=tf.AggregationMethod.ADD_N)
            with tf.train.MonitoredTrainingSession(master=server.target,
                                                    is_chief=(FLAGS.task_index == 0),
                                                    checkpoint_dir="./checkpoint_dir",
                                                    hooks=hooks) as mon_sess:
                while not mon_sess.should_stop():
                    batch_images, batch_labels = mnist.train.next_batch(32)
                    _, ls, step = mon_sess.run([train_op, loss, global_step],
                                               feed_dict={images: batch_images, labels: batch_labels})
                    if step % 100 == 0:
                        print("Train step %d, loss %f" % (step, ls))


if __name__ == "__main__":
    tf.app.run()

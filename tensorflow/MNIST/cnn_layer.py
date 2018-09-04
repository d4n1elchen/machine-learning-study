import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

## Here we'll create an estimator
def cnn_model_fn(features, labels, mode):

    # Input layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Conv layer 1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )
    # [-1, 28, 28, 32]

    # Pool layer 1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    # [-1, 14, 14, 32]

    # Conv layer 2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )
    # [-1, 14, 14, 64]

    # Pool layer 2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    # [-1, 7, 7, 64]

    # Dense
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(
        inputs=pool2_flat,
        units=1024,
        activation=tf.nn.relu
    )
    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.4,
        training=(mode == tf.estimator.ModeKeys.TRAIN)
    )

    # Logits (output)
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor") # Set name for PREDICT
    }

    # PREDICT mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # get loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits) # no need to apply softmax??

    # TRAIN mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # EVAL mode
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels,
                predictions=predictions["classes"]
            )
        }
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    raise Exception("Unknown ModeKey: " + mode)

def main(unused_argv):

    # Load data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Create the Estimator
    cnn = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir="mnist_cnn_model"
    )

    # Set up logging
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log,
        every_n_iter=50
    )

    # Train
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True
    )
    cnn.train(
        input_fn=train_input_fn,
        steps=200000,
        hooks=[logging_hook]
    )

    # Eval
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1, # what's this?
        shuffle=False
    )
    eval_results = cnn.evaluate(input_fn=eval_input_fn)
    print(eval_results)

if __name__ == "__main__":
    tf.app.run()

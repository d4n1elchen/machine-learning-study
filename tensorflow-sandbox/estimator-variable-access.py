import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

x = {"x": [1, 13, 26, 37, 47, 67, 49]}
y = [1,  1,  0,  1,  1,  1,  1]

my_feature_columns = [tf.feature_column.numeric_column(key="x")]

def my_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    return dataset.shuffle(100).repeat().batch(3).make_one_shot_iterator().get_next()

classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=[10],
        n_classes=2)

classifier.train(input_fn=my_input_fn, steps=500)

print(tf.global_variables())
print(tf.local_variables())
print(tf.trainable_variables())

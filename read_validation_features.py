import os
import glob
import tensorflow as tf
import numpy as np

def read_features(feature_dir='/home/didelson/Pictures/DataSets/ImageNet/Features_fc1_v4',
                  max_features=None,
                  dataset_split_name='validation'):

    if dataset_split_name is None:
        split_name_list = ['validation', 'train']
    else:
        split_name_list = [dataset_split_name]

    feature_vector = []
    label = []
    example = tf.train.Example()
    for split_name in split_name_list:
        for fn in glob.glob(os.path.join(feature_dir, '%s*_data.csv' % split_name)):
            feature_label = np.genfromtxt(fn.replace('_data', '_labels'), dtype=int, delimiter=',')
            feature_data = np.genfromtxt(fn, dtype=np.float32, delimiter=',')
            _, label_to_text_map = gel_label_text(feature_dir)
            feature_text = label_to_text_map[feature_label]

            print(feature_text)
            print(feature_data.shape)

            #stack arrays:
            #tf.stack(feature_vector)

            '''if os.path.isfile(fn):                
                for serialized_example in tf.python_io.tf_record_iterator(fn):
                    example.ParseFromString(serialized_example)
                    f=example.features.feature
                    # traverse the Example format to get data
                    #features = tf.parse_single_example(serialized_example, features={
                    #    'label': tf.FixedLenFeature([], tf.string),
                    #    'feature_vector': tf.FixedLenFeature([], tf.string),
                    #})

                    #label = tf.decode_raw(features['label'], tf.int64)
                    #vec_raw = tf.decode_raw(features['feature_vector'], tf.float32)
                    #vec_raw.set_shape(4096)
                    #feature_vector.append(vec_raw)
                    v = f['feature_vector'].float_list.value[16383]
                    img_1d = np.fromstring(v, dtype=np.float32)
                    reconstructed_img = img_1d.reshape(4096)

                    #feature_vec = example.features.feature['feature_vector'].float_list.value[1]
                    #feature_vec
                    #feature_vector.append(feature_vec)
                    #label.append(example.features.feature['label'].int64_list.value[0])'''

    #feature_vector = tf.stack(feature_vector)
    #sess = tf.get_default_session()
    #with tf.Session():
    #    result = feature_vector.eval()
    #print(feature_vector)
    #print(label)

# helper function if text labels is required:
def gel_label_text(feature_dir):

    labels_filename = os.path.join(feature_dir, 'labels.txt')

    with tf.gfile.Open(labels_filename, 'rb') as f:
       lines = f.read().decode()
    lines = lines.split('\n')
    lenn = len(lines) - 1
    lines = filter(None, lines)

    labels_to_class_names_dict = {}
    labels_to_class_names_array = np.empty(lenn, dtype=object)
    for line in lines:
       index = line.index(':')
       labels_to_class_names_dict[int(line[:index])] = line[index + 1:]
       labels_to_class_names_array[int(line[:index])] = line[index + 1:]

    return labels_to_class_names_dict, labels_to_class_names_array

if __name__ == '__main__':
    read_features()

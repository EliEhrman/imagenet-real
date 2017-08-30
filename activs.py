
from __future__ import print_function
import os
import sys
import glob
import math
import logging
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from time import gmtime, strftime

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_bool('debug', False,
					 'call tfdbg ')
tf.flags.DEFINE_bool('heavy', True,
					 'run on a serious GPGPU ')
tf.flags.DEFINE_bool('use_log_file', False,
					 'send output to pylog.txt ')
tf.flags.DEFINE_float('nn_lrn_rate', 0.01,
					 'base learning rate for nn ')
tf.flags.DEFINE_string('save_dir', '/tmp/savedmodels',
					   'directory to save model to. If empty, dont save')
tf.flags.DEFINE_float('fine_thresh', 0.00,
					 'once error drops below this switch to 50% like pairs ')

# files_dir = '../../data/imagenet/real'
files_dir = '../../data/imglabels'

if FLAGS.heavy:
	c_src_key_len = 4096
	c_target_key_len = 40
	c_hidden_dim = 1024
	c_batch_size = 256 # 256
	c_num_steps = 100000 # 10000
	c_files_str = 'validation*_data.csv'
	c_k_src_key = 4 # number of random pairs to find for each random key chosen. Must be an interger factor of c_rsize
	c_rsize = c_batch_size * c_batch_size / 32 # c_batch_size * (c_k_src_key * 2)
	c_eval_db_factor = 4 # what fraction of db to use for evaluation
	c_limit_num_files = 100
else:
	c_src_key_len = 4096
	c_hidden_dim = 40
	c_target_key_len = 20
	c_batch_size = 15
	c_rsize = 64
	c_num_steps = 100000
	c_files_str = 's*_data.csv'
	c_k_src_key = 4
	c_eval_db_factor = 1
	c_limit_num_files = 20

c_files_str = '*'
c_test_pc = 0.005
cb_do_validation = True

# function created to insert stops in the debugger. Create a stop/breakpoint by inserting a line like
# 				sess.run(t_for_stop)
def stop_reached(datum, tensor):
	if datum.node_name == 't_for_stop': # and tensor > 100:
		return True
	return False

t_for_stop = tf.constant(5.0, name='t_for_stop')

def t_repeat(t, rep_axis, tile_vec, name=None):
	return tf.tile(tf.expand_dims(t, rep_axis), tile_vec, name=name)

def build_nn_full(name_scope, t_nn_x, b_reuse):
	with tf.variable_scope('nn', reuse=b_reuse):
		v_W = tf.get_variable('v_W', shape=[c_src_key_len, c_hidden_dim], dtype=tf.float32,
							  initializer=tf.random_normal_initializer(stddev=1.0 / float(c_src_key_len * c_hidden_dim)))
		v_b1 = tf.get_variable('v_b1', shape=[c_hidden_dim], dtype=tf.float32,
							   initializer=tf.random_normal_initializer(stddev=1.0 / float(c_hidden_dim)))
		v_W2 = tf.get_variable('v_W2', shape=[c_hidden_dim, c_target_key_len], dtype=tf.float32,
							  initializer=tf.random_normal_initializer(stddev=1.0 / float(c_hidden_dim * c_target_key_len)))
		v_b2 = tf.get_variable('v_b2', shape=[c_target_key_len], dtype=tf.float32,
							   initializer=tf.random_normal_initializer(stddev=1.0 / float(c_target_key_len)))

	with tf.name_scope(name_scope):
		t_fc1 = tf.nn.bias_add(tf.matmul(t_nn_x, v_W), v_b1, name='t_fc1')
		t_nonlin1 = tf.nn.relu(t_fc1, name='t_nonlin1')
		t_fc2 = tf.nn.bias_add(tf.matmul(t_nonlin1, v_W2), v_b2, name='t_fc2')
		t_y = tf.nn.l2_normalize(t_fc2, dim=1, name='t_y')

	return t_y

def build_nn(name_scope, t_nn_x, b_reuse):
	with tf.variable_scope('nn', reuse=b_reuse):
		v_W = tf.get_variable('v_W', shape=[c_src_key_len, c_target_key_len], dtype=tf.float32,
							  initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0 / float(c_src_key_len * c_target_key_len)))

	with tf.name_scope(name_scope):
		t_y = tf.nn.l2_normalize(tf.matmul(t_nn_x, v_W), dim=1, name='t_y')

	return v_W, t_y


logger = logging.getLogger('imagenet vgg dimred')
# formatter = logging.Formatter('[%(levelname)s] %(message)s')

if FLAGS.use_log_file:
	# logging.basicConfig(filename='pylog.txt', level=logging.DEBUG)
	fh = logging.FileHandler('pylog.txt')
	fh.setLevel(logging.DEBUG)
	logger.addHandler(fh)
else:
	ch = logging.StreamHandler(stream=sys.stdout)
	# ch.setFormatter(formatter)
	ch.setLevel(logging.DEBUG)
	logger.addHandler(ch)

logger.setLevel(logging.DEBUG)
logger.info('Starting at: %s', strftime("%Y-%m-%d %H:%M:%S", gmtime()))

# split_name = 'validation'
# # for fn in glob.glob(os.path.join(files_dir, '%s*_data.csv' % split_name)):
# feature_data = np.ndarray([0, c_src_key_len], dtype=np.float32)

feature_label = []

for ifn, fn in enumerate(glob.glob(os.path.join(files_dir, c_files_str))):
	if ifn >= c_limit_num_files:
		break
	fnl = os.path.basename(fn)
	feature_label.append(fnl)
	# feature_label = np.genfromtxt(fn.replace('_data', '_labels'), dtype=int, delimiter=',')
	feature_data_raw = np.genfromtxt(fn, dtype=np.float32, delimiter=',')
	norm = np.linalg.norm(feature_data_raw, axis=1, keepdims=True)
	feature_data_part = feature_data_raw / norm
	[feature_label.append(fnl) for i in range(feature_data_part.shape[0]-1)]
	if ifn == 0:
		feature_data = feature_data_part
	else:
		feature_data = np.concatenate((feature_data, feature_data_part))
	# src_key_len = feature_data.shape[1]
	# break # for now, process just one file


# Determine how many records we read from the files
numrecs = feature_data.shape[0]
# shuffle the records
sufflle_stick = np.arange(numrecs)
np.random.shuffle(sufflle_stick)
feature_data = np.take(feature_data, sufflle_stick,axis=0)
feature_label = np.take(feature_label, sufflle_stick,axis=0)
# Create a variable to set the start location of the batch. Must be a variable so that multiple runs
# on that batch are actually the same batch. Shape=[]
v_batch_begin = tf.Variable(tf.random_uniform([], minval=0, maxval=numrecs - c_batch_size, dtype=tf.int32),
							trainable=False, name='v_batch_begin')
# Put all the data in a tensor. If the device can handle it, this means no data crosses the IO boundry.
# If not, no loss. Shape=[numrecs, c_src_key_len]
v_data = tf.Variable(tf.constant(feature_data), trainable=False, name='v_data')
t_x = tf.slice(input_=v_data, begin=[v_batch_begin, 0], size=[c_batch_size, c_src_key_len], name='t_x')
v_r1 = tf.Variable(tf.random_uniform([c_rsize], minval=0, maxval=c_batch_size-1, dtype=tf.int32),
				   trainable=False, name='v_r1')
v_r2 = tf.Variable(tf.random_uniform([c_rsize], minval=0, maxval=c_batch_size-1, dtype=tf.int32),
				   trainable=False, name='v_r2')


num_src_keys = c_rsize / (c_k_src_key * 2)
t_src_keys_idxs = tf.random_uniform([num_src_keys], minval=0, maxval=c_batch_size-1, dtype=tf.int32, name='t_src_keys_idxs')
t_src_keys = tf.gather(t_x, t_src_keys_idxs, name='t_src_keys')
t_src_key_cds = tf.matmul(t_src_keys, t_x, transpose_b=True, name='t_src_key_cds')
t_src_key_best_cds, t_src_key_best_idxs = tf.nn.top_k(t_src_key_cds, c_k_src_key, sorted=True, name='t_src_key_best_idxs')
op_r1 = tf.assign(v_r1,
				  tf.reshape(t_repeat(t_src_keys_idxs, [-1], [1, c_k_src_key * 2], 'src_keys_idxs_br'),
							 [-1]),
				  name='op_r1')
t_non_k_idxs = tf.random_uniform([num_src_keys, c_k_src_key], minval=0, maxval=c_batch_size-1, dtype=tf.int32, name='t_non_k_idxs')
t_pair_idxs = tf.concat([t_src_key_best_idxs, t_non_k_idxs], 1, name='t_pair_idxs')
op_r2 = tf.assign(v_r2,
				  tf.reshape(t_pair_idxs, [-1]),
				  name='op_r2')

t_x1 = tf.gather(t_x, v_r1, name='t_x1')
t_x2 = tf.gather(t_x, v_r2, name='t_x2')
# t_y = tf.matmul(t_x, tf.clip_by_value(v_W, 0.0, 10.0), name='t_y') # + b
v_W, t_y = build_nn('main', t_x, b_reuse=False)

t_y1 = tf.gather(t_y, v_r1, name='t_y1')
t_y2 = tf.gather(t_y, v_r2, name='t_y2')

t_cdx = tf.reduce_sum(tf.multiply(t_x1, t_x2), axis=1, name='t_cdx')
t_cdy = tf.reduce_sum(tf.multiply(t_y1, t_y2), axis=1, name='t_cdy')
t_err = tf.reduce_mean((t_cdx - t_cdy) ** 2, name='t_err')
op_train_step = tf.train.AdagradOptimizer(FLAGS.nn_lrn_rate).minimize(t_err, name='op_train_step')

db_size = int(float(numrecs/ c_eval_db_factor) * (1.0 - c_test_pc))
test_size = int(float(numrecs / c_eval_db_factor) * c_test_pc)
t_x_db = tf.slice(input_=v_data, begin=[0, 0],
				  size=[db_size, c_src_key_len],
				  name='t_x_db')
t_x_test = tf.slice(input_=v_data,
					begin=[db_size, 0],
					size=[test_size, c_src_key_len], name='t_x_test')
t_eval_x_cds =  tf.matmul(t_x_test, t_x_db, transpose_b=True, name='t_eval_x_cds')
_, t_eval_best_x_idxs = tf.nn.top_k(t_eval_x_cds, c_k_src_key, sorted=True, name='t_eval_best_x_idxs')
# t_x_db_best = tf.gather_nd(t_x_db, t_eval_best_idxs, name='t_x_db_best')
_, t_y_db = build_nn('db', t_x_db, b_reuse=True)
_, t_y_test = build_nn('test', t_x_test, b_reuse=True)
t_eval_y_cds =  tf.matmul(t_y_test, t_y_db, transpose_b=True, name='t_eval_y_cds')
_, t_eval_best_y_idxs = tf.nn.top_k(t_eval_y_cds, c_k_src_key, sorted=True, name='t_eval_best_y_idxs')

logger.info('num image records: %d', numrecs)
logger.info('learning rate: %f', FLAGS.nn_lrn_rate)
logger.info('target key size: %d', c_target_key_len)
logger.info('hidden dim: %d', c_hidden_dim)
logger.info('batch size: %d', c_batch_size)
logger.info('c_rsize: %d', c_rsize)
logger.info('c_k_src_key: %d', c_k_src_key)
logger.info('saving model to dir %s', FLAGS.save_dir)
logger.info('db size for eval: %d', db_size)
logger.info('num eval queries: %d', test_size)
logger.info('threshold for switch to fine: %f', FLAGS.fine_thresh)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver({"W":v_W}, max_to_keep=10)
ckpt = None
if FLAGS.save_dir:
	ckpt = tf.train.get_checkpoint_state(FLAGS.save_dir)
if ckpt and ckpt.model_checkpoint_path:
	logger.info('restoring from %s', ckpt.model_checkpoint_path)
	saver.restore(sess, ckpt.model_checkpoint_path)

if FLAGS.debug:
	sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type="curses")
	sess.add_tensor_filter("stop_reached", stop_reached)

losses = []

LP_COARSE = 0
LP_FINE = 1
learn_phase = LP_COARSE
print('Dont leave this here!!!!!!!!!!!!')
learn_phase = LP_FINE

for step in range(c_num_steps+1):
	if learn_phase == LP_COARSE:
		sess.run(tf.variables_initializer([v_batch_begin, v_r1, v_r2]))
	elif learn_phase == LP_FINE:
		sess.run(tf.variables_initializer([v_batch_begin]))
		sess.run([op_r1, op_r2])
	# logger.info(sess.run([t_src_key_cds, t_src_key_best_cds]))
	if step == 0:
		errval = math.sqrt(sess.run(t_err))
		r_W = sess.run(v_W)
		logger.info('Starting error: %f', errval)
	else:
		if step % (c_num_steps / 100) == 0:
			errval = np.mean(losses)
			losses = []
			logger.info('step: %d: error: %f', step, errval)
	if step % (c_num_steps / 100) == 0:
		if learn_phase == LP_COARSE and errval < FLAGS.fine_thresh:
			logger.info("Switching to fine learning!")
			learn_phase = LP_FINE
			# redo selecting pairs for v_r1 and v_r2 because we didn't start the iter in this phase
			sess.run([op_r1, op_r2])
		if saver and FLAGS.save_dir:
				saved_file = saver.save(sess,
										os.path.join(FLAGS.save_dir, 'model.ckpt'),
										step)
		sess.run(t_for_stop)
		r_best_x, r_best_y = sess.run([t_eval_best_x_idxs, t_eval_best_y_idxs])
		num_hits = 0
		# logger.info(r_W)
		if cb_do_validation:
			for iq, q_idxs in enumerate(r_best_y):
				target = feature_label[db_size + iq]
				x_labels = np.take(feature_label, r_best_x[iq]).tolist()
				y_labels = np.take(feature_label, r_best_y[iq]).tolist()
				logger.info([target, 'x:', r_best_x[iq], 'y:', r_best_y[iq], 'x labels:', x_labels, 'y labels:', y_labels])
				for idx in q_idxs:
					num_hits += np.count_nonzero(r_best_x[iq] == idx)
			logger.info('Eval. how many of ys found in x: %f pct ', 100.0 * float(num_hits) / float(len(r_best_x) * len(r_best_x[0])))

	outputs = sess.run([t_err, op_train_step])
	losses.append(math.sqrt(outputs[0]))

print('done')

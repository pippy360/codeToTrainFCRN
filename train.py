import tensorflow as tf
import os
from PIL import Image
import numpy as np
import math
import time
import datetime

from network import network, fcrn


CHECKPOINT_DIR = './output/ch'
SUMMARY_DIR = 'output/train'
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'

IMAGE_HEIGHT = 228
IMAGE_WIDTH = 304
TARGET_HEIGHT = 128
TARGET_WIDTH = 160
BATCH_SIZE = 10

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 500
NUM_EPOCHS_PER_DECAY = 1
INITIAL_LEARNING_RATE = 0.0000001
LEARNING_RATE_DECAY_FACTOR = 0.9
MOVING_AVERAGE_DECAY = 0.999999


def getFilenameQueuesFromCSVFile( csvFilePath ):
    filename_queue = tf.train.string_input_producer([csvFilePath], shuffle=False)
    reader = tf.TextLineReader()
    _, serialized_example = reader.read(filename_queue)
    filename, depth_filename = tf.decode_csv(serialized_example, [["path"], ["annotation"]])
    return filename, depth_filename

def csv_inputs(image_filename, depth_filename, batch_size, imageSize, depthImageSize):

    # input
    image = tf.read_file(image_filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, imageSize)

    # target
    depth = tf.read_file(depth_filename)
    depth = tf.image.decode_png(depth, channels=3)#fixme: change to 1 channel
    depth = tf.image.resize_images(depth, depthImageSize)
    depth = tf.slice(depth, [0,0, 0], [-1,-1,1])

    # resize
    invalid_depth = tf.sign(depth)
    # generate batch
    images, depths, invalid_depths, filenames = tf.train.batch(
        [image, depth, invalid_depth, image_filename],
        batch_size=batch_size,
        num_threads=4,
        capacity= 50 + 3 * batch_size,
    )
    return images, depths, invalid_depths, filenames


#FIXME: REMOVE THIS 
IMAGE_HEIGHT = 228
IMAGE_WIDTH = 304
TARGET_HEIGHT = 128
TARGET_WIDTH = 160
BATCH_SIZE = 8

def loss_scale_invariant_l2_norm(logits, depths, invalid_depths):
    logits_flat = tf.reshape(logits, [-1, TARGET_HEIGHT*TARGET_WIDTH])
    depths_flat = tf.reshape(depths, [-1, TARGET_HEIGHT*TARGET_WIDTH])
    invalid_depths_flat = tf.reshape(invalid_depths, [-1, TARGET_HEIGHT*TARGET_WIDTH])

    predict = tf.multiply(logits_flat, invalid_depths_flat)
    target = tf.multiply(depths_flat, invalid_depths_flat)
    d = tf.subtract(predict, target)
    square_d = tf.square(d)
    sum_square_d = tf.reduce_sum(square_d, 1)
    sum_d = tf.reduce_sum(d, 1)
    sqare_sum_d = tf.square(sum_d)
    cost = tf.reduce_mean(sum_square_d / TARGET_HEIGHT*TARGET_WIDTH - 0.5*sqare_sum_d / math.pow(TARGET_HEIGHT*TARGET_WIDTH, 2))
    tf.add_to_collection('losses', cost)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def loss_l2_norm(logits, depths, invalid_depths):
    logits_flat = tf.reshape(logits, [-1, TARGET_HEIGHT*TARGET_WIDTH])
    depths_flat = tf.reshape(depths, [-1, TARGET_HEIGHT*TARGET_WIDTH])

    d = tf.subtract(logits_flat, depths_flat)
    return tf.nn.l2_loss(d)

def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.
  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.
  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op



def train(total_loss, global_step):
  """Train CIFAR-10 model.
  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.
  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
  print('decay_steps')
  print(decay_steps)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  lr = INITIAL_LEARNING_RATE
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer( lr )
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)
  return apply_gradient_op
  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  with tf.control_dependencies([apply_gradient_op]):
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    variables_to_restore = variable_averages.variables_to_restore()

  return variables_averages_op, variables_to_restore

def restore(sess, switch=False):
    #can we be sure we'll have these moving averages??????
    #TODO: handle case when no checkpoint
    # Restore the moving average version of the learned variables for eval.
    # Use to load from ckpt file
    model_data_path = tf.train.latest_checkpoint( CHECKPOINT_DIR )
    if model_data_path is None:
	return None
    print("model_data_path")
    print("model_data_path")
    print(model_data_path)
    #model_data_path = './network2/checkpoint/NYU_FCRN.ckpt'
    saver = tf.train.Saver()
    saver.restore(sess, model_data_path)

    # Assuming model_checkpoint_path looks something like:
    #   /my-favorite-path/cifar10_train/model.ckpt-0,
    # extract global_step from it.
    global_step = model_data_path.split('/')[-1].split('-')[-1]

    print('checkpoint loaded with global_step: ' + str(global_step))
    return 1

def getInference( images ):
    net = fcrn.ResNet50UpProj({'data': images}, 1, 1, False)
    return net.get_output()	

def saveImage(imageArr, outputFileLocation):
	formatted = ( ( imageArr[0,:,:,0] ) * 255 / np.max( imageArr[0,:,:,0] ) ).astype('uint8')
	img = Image.fromarray( formatted )
	img.save( outputFileLocation )

def addImageSummary():
	pass

def handleImageValues( images_raw, depths_raw ):
	return tf.divide(images_raw, 255.0), tf.divide(depths_raw, 255.0)
	

def runIt(inputNetwork):
    with tf.Graph().as_default():
	global_step = tf.train.get_or_create_global_step()
        imageSize = ( IMAGE_HEIGHT, IMAGE_WIDTH )
        depthImageSize = ( TARGET_HEIGHT, TARGET_WIDTH )
	filename, depth_filename = getFilenameQueuesFromCSVFile( TRAIN_FILE )
        images_raw, depths_raw, invalid_depths, filenames = csv_inputs( filename, depth_filename, BATCH_SIZE, imageSize=imageSize, depthImageSize=depthImageSize )
	images, depths = handleImageValues( images_raw, depths_raw )
	addImageSummary()
        logits = getInference(images)
        tf.summary.image('input_images', logits, max_outputs=3)
        #loss_op = loss_scale_invariant_l2_norm(logits, depths, invalid_depths)
        loss_op = loss_l2_norm( logits, depths, invalid_depths )

  	train_op = train( loss_op, global_step )

        init = tf.global_variables_initializer()
	with tf.Session() as sess:

		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter( SUMMARY_DIR, sess.graph )

		checkCkpt = None #restore(sess, restoreVar)
		
		coord = tf.train.Coordinator()
		try:
			threads = []
			for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
				threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
			if checkCkpt is None:
				sess.run([init])
				
			print("init run")

			for i in range(1000000):
				if not i % 100 == 0 and not i % 10 == 0  :
					summary, _ = sess.run([merged, train_op])
					train_writer.add_summary(summary, i)
				elif not i % 100 == 0 and i % 10 == 0 :
					summary, _, loss = sess.run([merged, train_op, loss_op])
					train_writer.add_summary(summary, i)
					print( "step: " + str(i) + "\tloss: " + str(loss) )
				else:
					_, loss, outputDepth, testDepth, image, filenames2 = sess.run([train_op, loss_op, logits, depths, images, filenames])
					print( "step: " + str(i) + "\tloss: " + str(loss) )
					
					pred = outputDepth
					#print("np.max(pred[0,:,:,0])):")
					#print( np.max(pred[0,:,:,0]) )
					formatted = ((pred[0,:,:,0]) * 255 / np.max(pred[0,:,:,0])).astype('uint8')
					img = Image.fromarray(formatted)
					img.save("./output/output.png")

					pred = testDepth 
					#print("np.max(pred[0,:,:,0])):")
					#print( np.max(pred[0,:,:,0]) )
					formatted = ((pred[0,:,:,0]) * 255 / np.max(pred[0,:,:,0])).astype('uint8')
					img = Image.fromarray(formatted)
					img.save("./output/outputDepth.png")
					#print('image[0]')
					#print(image[0])
					#print(image[0].shape)
					image2 = np.zeros((512,512,3), 'uint8')
					#print(image.shape)
					img = Image.fromarray(image[0].astype('uint8'))
					img.save("./output/outputOrgImage.jpg")
					#print('filenames')
					#print(filenames2)
					saver = tf.train.Saver()
					saver.save(sess, './output/ch/ch')
		except Exception as e:
			coord.request_stop(e)
			print(e)
			raise e
		print("done")


def main(argv=None):  # pylint: disable=unused-argument
	if not os.path.exists(CHECKPOINT_DIR):
		os.makedirs(CHECKPOINT_DIR)
	inputNetwork = network
	runIt(inputNetwork)

main()

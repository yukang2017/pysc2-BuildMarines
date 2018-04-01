from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pysc2.env import environment

import time
import tensorflow as tf

def run_loop(FLAGS, agents, env, max_frames = 0):
  """A run loop to have agents and an environment interact."""
  if FLAGS.training:
      PARALLEL = FLAGS.parallel
      DEVICE = ['/gpu:'+dev for dev in FLAGS.device.split(',')]
  else:
      PARALLEL = 1
      DEVICE = ['/cpu:0']
  LOG = FLAGS.log_path+FLAGS.map+'/'+FLAGS.net
  SNAPSHOT = FLAGS.snapshot_path+FLAGS.map+'/'+FLAGS.net

  num_echos = -1

  while True:
#      try:
    start_time = time.time()
    num_frames = 0
    num_echos += 1
    sum_reward = 0

    if(num_echos==1):
        for agent in agents:
            for i in range(PARALLEL):
                agent.build_model(i > 0, DEVICE[i % len(DEVICE)], FLAGS.net)
                    
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            summary_writer = tf.summary.FileWriter(LOG)
            for i in range(PARALLEL):
                agents[i].setup(sess, summary_writer)
                
            agent.initialize()
            if not FLAGS.training or FLAGS.continuation:
                global COUNTER
                COUNTER = agent.load_model(SNAPSHOT)
                    
    timesteps = env.reset()
    for a in agents:
        a.reset()
    replay_buffer = []
    while True:
        num_frames += 1
        last_timesteps = timesteps
        if num_echos==0:
            actions = [agent.scriptAgent.step(timestep) for agent, timestep in zip(agents, timesteps)]
        else:
            actions = [agent.step(timestep) for agent, timestep in zip(agents, timesteps)]
          #print('action=',actions)
        timesteps = env.step(actions)
        sum_reward += timesteps[0].reward
          
          # Only for a single player!
        if FLAGS.training:
            recorder = [last_timesteps[0], actions[0], timesteps[0]]
            replay_buffer.append(recorder)
        is_done = (num_frames >= max_frames) or timesteps[0].last() or (timesteps[0].step_type == environment.StepType.LAST)
        if is_done:
          print('After yield, done!')  
          break

    yield replay_buffer, num_echos, is_done

#      except KeyboardInterrupt:
#        pass
    elapsed_time = time.time() - start_time
    print("Echo %d : rewards %d , tooks %.3f seconds for %s steps: %.3f fps" % (
                num_echos, sum_reward, elapsed_time, num_frames, num_frames / elapsed_time))
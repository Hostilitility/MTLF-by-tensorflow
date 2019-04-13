'''
MTLF algorithm for regression, written by Yu Ye
The algorithm was proposed by:
[1] Xu Y, Pan S J, Xiong H, et al. A unified framework for metric transfer learning[J].
    IEEE Transactions on Knowledge and Data Engineering, 2017, 29(6): 1158-1171.
Only the classification edition of MTLF on Matlab is given by the author.So we implement the
algorithm on tensorflow, and replace SGD, the default optimization algorithm, with Adam.
'''

import tensorflow as tf
import numpy as np
import time
from densratio import densratio

def MTLF(source_x, source_y, target_x, target_y, test_x, learning_rate = 0.01,
         alpha = 1, lamda = 1, beta = 0.01 , rho = 1, sigma = 1,max_step = 31):
    '''
    :param source_x: (n_source,n_dim)
    :param source_y: (n_source,n_task)
    :param target_x: (n_target,n_dim)
    :param target_y: (n_target,n_task)
    :param test_x: (n_test,n_dim)
    :return: predict: (n_test,n_task)
    '''

    def compute_k_mat(X, A, sigma,n_sample):
        '''
        :param X: (n_sample,n_dim)
        :param A: (n_dim,n_dim)
        :return K: (n_sample, n_sample)
        '''
        # 还可以进一步加速
        K = [[]]
        for i in range(n_sample):
            K = tf.concat([K, tf.expand_dims(tf.matrix_diag_part(tf.matmul(
                tf.matmul(A, tf.tile(tf.expand_dims(X[i], 0), [n_sample, 1]) - X, transpose_b=True),
                tf.matmul(A, tf.tile(tf.expand_dims(X[i], 0), [n_sample, 1]) - X, transpose_b=True), True)), 0)],
                          int(i == 0))
        K = tf.exp(-K / sigma / sigma)
        return K

    epsilon = 1e-10

    (n_source,n_dim) = source_x.shape
    (n_target,n_task) = target_y.shape
    n_sample = n_source + n_target
    n_test = test_x.shape[0]
    all_target_x = np.row_stack([target_x,test_x])
    dense_result = densratio(all_target_x,source_x)
    w0 = dense_result.compute_density_ratio(source_x)
    w0 /= np.mean(w0)
    predict = np.zeros([n_test,n_task])

    #输入节点
    Source_x = tf.placeholder(dtype=tf.float32,shape = [None,n_dim])
    Source_y = tf.placeholder(dtype=tf.float32,shape = [None,])
    Target_x = tf.placeholder(dtype=tf.float32,shape = [None,n_dim])
    Target_y = tf.placeholder(dtype=tf.float32,shape=[None,])
    Test_x = tf.placeholder(dtype=tf.float32,shape = [None,n_dim])

    Union_x = tf.concat([Source_x,Target_x],axis=0)
    Union_y = tf.concat([Source_y,Target_y],axis=0)

    Max_y = tf.reduce_max(Union_y)
    Union_y = Union_y/Max_y

    #中间节点
    A = tf.Variable(tf.eye(n_dim))
    Source_w = tf.Variable(w0,dtype=tf.float32)
    Union_w = tf.concat([Source_w,tf.ones(n_target)],0)

    #计算loss
    loss1 = tf.trace( tf.matmul(A,A,transpose_a=True) )
    loss2 = tf.norm( Source_w - w0 + epsilon)
    K = compute_k_mat(Union_x,A,sigma,n_sample)
    train_pred = []
    for i in range(n_sample):
        train_pred = tf.concat([train_pred,[tf.reduce_sum(K[i] * Union_y)/tf.reduce_sum(K[i])]],0)
    loss3 = tf.reduce_sum((train_pred - Union_y)**2 * Union_w)
    loss4 = (tf.reduce_sum(Source_w)-n_source)**2
    loss5 = tf.reduce_sum(tf.maximum(0.,-Source_w))**2
    loss = alpha*loss1 + lamda*loss2 + beta*loss3 + rho*(loss4+loss5)

    #输出节点
    Pred = []
    for i in range(n_test):
        Kernel = tf.matmul(
            tf.matmul(A,tf.tile(tf.expand_dims(Test_x[i],0),[n_sample,1])-Union_x,transpose_b = True) ,
            tf.matmul(A,tf.tile(tf.expand_dims(Test_x[i],0),[n_sample,1])-Union_x,transpose_b = True) ,True )
        Kernel = tf.exp(-Kernel/sigma/sigma)
        Pred = tf.concat([Pred, [tf.reduce_sum(Kernel * Union_y) / tf.reduce_sum(Kernel)]], 0)
    Pred *= Max_y

    loss_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = loss_optimizer.minimize(loss)

    time_start = time.time()
    for i in range(n_task):
        print('MTLF %d/%d'%(i,n_task))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for step in range(max_step):
                step_loss = sess.run(loss,feed_dict={
                    Source_x:source_x,Source_y:source_y[:,i],
                    Target_x:target_x,Target_y:target_y[:,i],
                    Test_x:test_x
                })
                print('step : %d \t\ttrain loss : %f'%(step,step_loss))
                sess.run(train_op, feed_dict={
                    Source_x: source_x, Source_y: source_y[:, i],
                    Target_x: target_x, Target_y: target_y[:, i],
                    Test_x: test_x
                })
            final_A,final_w = sess.run([A,Source_w])
            predict[:,i] = sess.run(Pred,feed_dict={
                    Source_x:source_x,Source_y:source_y[:,i],
                    Target_x:target_x,Target_y:target_y[:,i],
                    Test_x:test_x
                })
    time_end = time.time()
    print('MTLF average time = %.4f'%((time_end-time_start)/n_task))
    return predict
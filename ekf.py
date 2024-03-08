import numpy as np

#该函数用于执行预测
def predict(sta,u,cov,Rt):
    """
    :param sta:当前状态的状态
    :param u:控制信息（平移、旋转）
    :param cov:当前状态协方差matrix
    :param Rt:噪声协方差矩阵
    """
    n = len(sta)

    #定义运动模型函数f(sta,u)——计算当前运动对应的状态变化motion
    [dtrans, drot1, drot2] = u
    motion = np.array([[dtrans * np.cos(sta[2][0] + drot1)],
                       [dtrans * np.sin(sta[2][0] + drot1)],
                       [drot1 + drot2]])
    F = np.array(np.eye(3),np.zeros((3,n-3)),axis=1)

    #新状态的预测
    sta_new = sta + (F.T).dot(motion)

    #雅各比矩阵
    J = np.array([[0,0,-dtrans*np.sin(sta[2][0]+drot1)],
               [0,0,dtrans*np.cos(sta[2][0]+drot1)],
               [0,0,0]])
    #状态变换的导数
    G = np.eye(n) + (F.T).dot(J).dot(F)

    #new协方差矩阵
    cov_new = G.dot(cov).dot(G.T) + (F.T).dot(Rt).dot(F)

    print('Predicted location\t x: {0:.2f} \t y: {1:.2f} \t theta: {2:.2f}'.format(sta_new[0][0], sta_new[1][0],
                                                                                   sta_new[2][0]))
    return sta_new, cov_new

#该函数用于执行更新步骤
def update(sta,cov,obs,c_prob,Qt):
    N = len(sta)

    #对每一个数据进行迭代（观测距离r、观测角度theta、地标编号j）
    for [r, theta, j] in obs:
        j = int(j)
        #检测landmark是否被检测过
        if cov[2 * j + 3][2 * j + 3] >= 1e6 and cov[2 * j + 4][2 * j + 4] >= 1e6:
            #把当前的landmark作为观测
            sta[2 * j + 3][0] = sta[0][0] + r * np.cos(theta + sta[2][0])
            sta[2 * j + 4][0] = sta[1][0] + r * np.sin(theta + sta[2][0])

        #landmark是否静止
        if c_prob[j] >= 0.5:
            #计算预期的观测值
            delta = np.array([sta[2 * j + 3][0] - sta[0][0], sta[2 * j + 4][0] - sta[1][0]])
            q = delta.T.dot(delta)
            sq = np.sqrt(q)
            z_theta = np.arctan2(delta[1], delta[0])
            z_hat = np.array([[sq], [z_theta - sta[2][0]]])

            #雅各比
            F = np.zeros((5, N))
            F[:3, :3] = np.eye(3)
            F[3, 2 * j + 3] = 1
            F[4, 2 * j + 4] = 1
            H_z = np.array([[-sq * delta[0], -sq * delta[1], 0, sq * delta[0], sq * delta[1]],
                            [delta[1], -delta[0], -q, -delta[1], delta[0]]], dtype='float')
            H = 1 / q * H_z.dot(F)

            #卡尔曼增益——融合预期观测和实际观测
            K = cov.dot(H.T).dot(np.linalg.inv(H.dot(cov).dot(H.T) + Qt))

            #根据卡尔曼增益和观测差异更新状态向量和协方差矩阵
            z_dif = np.array([[r], [theta]]) - z_hat
            z_dif = (z_dif + np.pi) % (2 * np.pi) - np.pi

            #输出更新后的机器人位置信息。
            sta = sta + K.dot(z_dif)
            cov = (np.eye(N) - K.dot(H)).dot(cov)

        print('Updated location\t x: {0:.2f} \t y: {1:.2f} \t theta: {2:.2f}'.format(sta[0][0], sta[1][0], sta[2][0]))
        return sta, cov, c_prob
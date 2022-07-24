# coding:utf8
import visdom
import time
import numpy as np
import matplotlib.pyplot as plt

class Visualizer(object):
    """
    封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`
    调用原生的visdom接口
    比如
    self.text('hello visdom')
    self.histogram(t.randn(1000))
    self.line(t.arange(0, 10),t.arange(1, 11))
    """

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env,use_incoming_socket=False, **kwargs)

        # 画的第几个数，相当于横座标
        # 保存（’loss',23） 即loss的第23个点
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        """
        修改visdom的配置
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        """
        一次plot多个
        @params d: dict (name,value) i.e. ('loss',0.11)
        """
        for k, v in d.items():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        """
        self.plot('loss',1.00)
        """
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def scatter_reach(self, name, d):
        """
        画三维散点图
        """
        label = []
        label.append(1)
        for i in range(1, len(d)-1):
            label.append(2)
        label.append(3)
        win = self.vis.scatter(
            X=np.array(d),
            Y=np.array(label),
            win=name,
            opts=dict(
                title=name,
                legend=['start', 'action', 'target'],
                xtickmin=0.2,
                xtickmax=0.7,
                xtickstep=0.05,
                ytickmin=-0.3,
                ytickmax=0.3,
                ytickstep=0.05,
                ztickmin=0.0,
                ztickmax=0.55,
                ztickstep=0.05,
                markersize=5,
                height=500,
                width=500,
                markercolor=np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]]),
            )
        )
        return win

    def scatter_updata(self, win, d):
        self.vis.scatter(
            X=[d],
            win=win,
            update='new',
            opts=dict(
                markersize=15,
                markercolor=np.array([[0, 0, 255]]),
            )
        )

    def scatter_a(self, name ,d):
        """
        画三维点线图
        """
        x = []
        y = []
        z = []
        ax = plt.gca(projection="3d")
        for i in range(len(d)):
            x.append(d[i][0])
            y.append(d[i][1])
            z.append(d[i][2])
        # x, y, z = [1, 1.5, 3], [1, 2.4, 3], [3.4, 1.4, 1]
        ax.scatter(x, y, z, c='r')
        ax.plot(x, y, z, color='r')
        plt.title(name)
        self.vis.matplot(plt)

    def img(self, name, img_, **kwargs):
        """
        self.img('input_img',t.Tensor(64,64))
        self.img('input_imgs',t.Tensor(3,64,64))
        self.img('input_imgs',t.Tensor(100,1,64,64))
        self.img('input_imgs',t.Tensor(100,3,64,64),nrows=10)

        ！！！don‘t ~~self.img('input_imgs',t.Tensor(100,64,64),nrows=10)~~！！！
        """
        self.vis.images(img_.cpu().numpy(),
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1,'lr':0.0001})
        """

        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        """
        自定义的plot,image,log,plot_many等除外
        self.function 等价于self.vis.function
        """
        return getattr(self.vis, name)

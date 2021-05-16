'''
Code referenced from:
https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
'''

from visualdl import LogWriter


__Author__ = 'Quanhao Guo'
__Date__ = '2021.05.03.16.32'


class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = LogWriter(logdir=log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""
        for i, img in enumerate(images):
            self.writer.add_image(tag='%s/%d' % (tag, i), img=img, step=step)


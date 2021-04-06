import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils import denormalize, bounding_box
from torchvision import transforms

def parse_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument("--plot_dir", type=str, required=True,
                     help="path to directory containing pickle dumps")
    arg.add_argument("--epoch", type=int, required=True,
                     help="epoch of desired plot")
    args = vars(arg.parse_args())
    return args['plot_dir'], args['epoch']


def main(plot_dir, epoch):

    # read in pickle files
    glimpses = pickle.load(
        open(plot_dir + "g_{}.p".format(epoch), "rb")
    )
    locations = pickle.load(
        open(plot_dir + "l_{}.p".format(epoch), "rb")
    )
    probas = pickle.load(
        open(plot_dir + "p_{}.p".format(epoch), "rb")
    )
    groundtruth = pickle.load(
        open(plot_dir + "y_{}.p".format(epoch), "rb")
    )

    glimpses = np.concatenate(glimpses)
    glimpses = np.moveaxis(glimpses, 1, -1)

    # grab useful params
    size = int(plot_dir.split('_')[2].split('x')[0])
    num_anims = len(locations)
    num_cols = glimpses.shape[0]
    img_shape = glimpses.shape[1]
    num_classes = probas[0].shape[1]

    print("size: {}\n num_anims: {}\n img_shape: {} \n num_classes: {}".format(size,num_anims,img_shape,num_classes))

    # denormalize coordinates
    coords = [denormalize(img_shape, l) for l in locations]

    fig, axs = plt.subplots(nrows=2, ncols=num_cols)
    # fig.set_dpi(200)

    inv_normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255])])
        
    # plot base image
    for j, ax in enumerate(axs.flat[:num_cols]):
        #img = np.moveaxis(glimpses[j], 0, -1)
        #img = glimpses[j]
        img = inv_normalize(glimpses[j])
        print(img.shape)
        img = np.transpose(img)
        img = np.swapaxes(img, 0, 1)
        
        ax.imshow(img) # cmap="Greys_r",
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    # plot probability distributions
    # for j, ax in enumerate(axs.flat[num_cols:]):
    #     ax.bar(range(1,num_classes+1), np.exp(probas[j][0]))[groundtruth[j]].set_color("g")
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    #     ax.set_ylim(top=1)


    def updateData(i):
        color = 'r'
        if i == 0:
            color = "g"
        else:
            color = "r"
        co = coords[i]
        for j, ax in enumerate(axs.flat[:num_cols]):
            for p in ax.patches:
                p.remove()
            # ugly fix. as not all patches are removed first time do it twice ¯\_(ツ)_/¯
            for p in ax.patches:
                p.remove()

            c = co[j]
            rect = bounding_box(
                #c[0], c[1], size, color
                c[1], c[0], size, color
            )
            rect2 = bounding_box(
                #c[0], c[1], size*3, color
                c[1], c[0], size*3, color
            )
            rect3 = bounding_box(
                #c[0], c[1], size*3*3, color
                c[1], c[0], size*3*3, color
            )
            ax.add_patch(rect)
            ax.add_patch(rect2)
            ax.add_patch(rect3)
        for j, ax in enumerate(axs.flat[num_cols:]):
            ax.clear()
            ax.bar(range(1,num_classes+1), np.exp(probas[j][i]))[groundtruth[j]].set_color("g")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_ylim(top=1)

    # animate
    anim = animation.FuncAnimation(
        fig, updateData, frames=num_anims, interval=500, repeat=True
    )
    plt.show()

    class LoopingPillowWriter(animation.PillowWriter):
        def finish(self):
            self._frames[0].save(
                self._outfile, save_all=True, append_images=self._frames[1:],
                duration=int(1000 / self.fps), loop=0)

    name = plot_dir + 'epoch_{}.gif'.format(epoch)
    anim.save(name, dpi=10, writer=LoopingPillowWriter(fps=1))
    # save as mp4
    # name = plot_dir + 'epoch_{}.mp4'.format(epoch)
    # anim.save(name, extra_args=['-vcodec', 'h264', '-pix_fmt', 'yuv420p'])


if __name__ == "__main__":
    args = parse_arguments()
    main(*args)

import os
import shutil


def sample_images(src_dir, tgt_dir, tgt_ipc):
    for file in sorted(os.listdir(src_dir)):
        if os.path.isdir(os.path.join(src_dir, file)):
            print(f"current class dir: {file}")
            images = sorted(
                [f for f in os.listdir(os.path.join(src_dir, file)) if f.endswith(('.jpg', '.png', '.jpeg'))])
            if len(images) >= tgt_ipc:
                os.makedirs(os.path.join(tgt_dir, file), exist_ok=True)
                sampled_images = images[:tgt_ipc]
                for sampled_image in sampled_images:
                    print(f"current image: {sampled_image}")
                    shutil.copy(os.path.join(src_dir, file, sampled_image), os.path.join(tgt_dir, file, sampled_image))
            else:
                print(f"source dir don't contain {tgt_ipc} images, skip..")
        else:
            print(f"current file {file} isn't a directory, skip..")


if __name__ == '__main__':
    source_dir = os.path.join("..", "Datasets", "generated_data", "syn_data", "SRe2Lplus_FD2_CUB_imsize224_09FC05_01SC4", "rec_res18_ipc5")  # Replace with your path
    target_ipc = 3
    target_dir = os.path.join("..", "Datasets", "generated_data", "syn_data", "SRe2Lplus_FD2_CUB_imsize224_09FC05_01SC4", f"rec_res18_ipc{target_ipc}")  # Replace with your path
    sample_images(source_dir, target_dir, target_ipc)

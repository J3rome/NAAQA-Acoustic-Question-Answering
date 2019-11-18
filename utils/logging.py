from tensorboardX import SummaryWriter


def create_tensorboard_writers(args, paths):
    # FIXME : What happen with test set? I guess we don't really care, we got our own visualisations for test run
    # Create tensorboard writer
    base_writer_path = '%s/%s/%s' % (
    args['tensorboard_folder'], paths["output_name"], paths["current_datetime_str"])

    # TODO : Add 'comment' param with more infos on run. Ex : Raw vs Conv
    return {
        'writers': {
            'train': SummaryWriter('%s/train' % base_writer_path),
            'val': SummaryWriter('%s/val' % base_writer_path)
        },
        'options': {
            'save_images': args['tensorboard_save_images'],
            'save_texts': args['tensorboard_save_texts']
        }
    }


def close_tensorboard_writers(tensorboard_writers):
    for key, writer in tensorboard_writers.items():
        writer.close()

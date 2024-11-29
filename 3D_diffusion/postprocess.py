import numpy as np
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', '-i',   default='./output/samples.npy')
    parser.add_argument('--category',   '-c',   default=None)
    parser.add_argument('--output_dir', '-o',   default='./output')
    opt = parser.parse_args()

    assert opt.category in ['airplane', 'chair', 'table']
    if opt.category == 'airplane':
        threshold = 0.07
    elif opt.category == 'chair':
        threshold = 0.07
    elif opt.category == 'table':
        threshold = 0.07
    else:
        raise AssertionError
        
    samples = np.load(opt.input_path)
    binary = (samples >= threshold)
    np.save(os.path.join(opt.output_dir, f"{opt.input_path.split('/')[-1].replace('.npy','')}_binary.npy"), binary)
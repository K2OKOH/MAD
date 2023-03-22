# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc
from datasets.cityscape import cityscape

from datasets.dgunionlable import dgunionlable

from datasets.coco import coco
from datasets.imagenet import imagenet
from datasets.vg import vg
from datasets.voc_setting_2d import voc_setting_2d

import numpy as np

# 1M2S
for year in ['2007', '2012']:
  for M_Set in            ['voc', 'cityscape', 'kitti', 'watercolor', 'clipart', 'sim10k', 'kitti', 'bdd100k', 'rain', 'foggy']:
    for S1_Set in         ['voc', 'cityscape', 'kitti', 'watercolor', 'clipart', 'sim10k', 'kitti', 'bdd100k', 'rain', 'foggy']:
      for S2_Set in       ['voc', 'cityscape', 'kitti', 'watercolor', 'clipart', 'sim10k', 'kitti', 'bdd100k', 'rain', 'foggy']:
        for type in       ['s1','s2','s3','s4','s5','s6','s7','s8','s10','s14','s15','s16','s17','s18','t','x1','fa','fs']:
          for split in    ['train','test','train_s','test_s','train_t','test_t','trainval','train_h', 'test_2k',  'test_200']:
            name = '{}_{}_{}_{}_{}_{}'.format(M_Set, S1_Set, S2_Set, type, year, split)
            __sets[name] = (lambda Main_Set=M_Set, type=type, split=split, year=year, Sub1_Set=S1_Set, Sub2_Set=S2_Set : voc_setting_2d(Main_Set, type, split, year, Sub1_Set, Sub2_Set))

for year in ['2007', '2012']:
    for testdataset in ['unionvoc', 'unioncityscape', 'kitti', 'watercolor', 'clipart', 'sim10k','kitti','bdd100k','fogycityscape']:
        for dataset in ['unionvoc', 'unioncityscape', 'kitti', 'watercolor', 'clipart', 'sim10k','kitti','bdd100k','fogycityscape']:
            for source in ['_s1','_s2','_s3','_s4','_s5','_s6','_s7','_s8','_s10','_s14','_s15','_s16','_s17','_s18','_single1','_single2','_single6','_s30','_s31','_s32','_t']:

                for split in ['train','test','trainval','train_h','mval','mtrain','test_2k']:
                    name = '{}_{}_{}'.format(dataset+source, year, split+testdataset)
                    __sets[name] = (lambda testdataset=testdataset, dataset=dataset, split=split, year=year, source=source: dgunionlable(testdataset, dataset, source, split, year))

def get_imdb(name):
  #print(name)
  #print(list_imdbs())
  #print(__sets['kitti_2007_train'])
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
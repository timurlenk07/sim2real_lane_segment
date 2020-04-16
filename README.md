# Right Lane segmentation in Duckietown environment
Repository for private project Duckietown at BME VIK (Budapesti Műszaki és Gazdaságtudományi Egyetem Villamosmérnöki és Informatikai Kar).

## Data generation for semantic segmentation in Duckietown environment
For all machine learning task data quality and quantity is of crucial importance.
On one hand more data helps the feature extractor to better generalize while data diversity ensures that the model has seen all kind of possible inputs during training.
Tipically, labelled databases are not publicly available for all kind of tasks and that is especially true for such new challenges sa the Duckietown competition.

Luckily, there is a publicly available open source simulator software that was designed for reinforcement learning tasks and that is free to use, that means with some modifications we were able to extract (or record) some runs that were saved both as an observation picture as well as labelled picture for semantic segmentation.
After some post-processing, observation and label videos were produced that helps in building some supervised or semi-supervised machine learning algorithm.

If we are interested in pictures only, not videos, then these can be taken apart into separate images.

For more information on how the data collection and preprocessing was done, please refer to [this wiki page](TODO site).

### Example training data
Data pair 1:

![Original](doc/res_readme/orig_1.jpg)
![Annotated](doc/res_readme/annot_1.jpg)

Data pair 2:

![Original](doc/res_readme/orig_2.jpg)
![Annotated](doc/res_readme/annot_2.jpg)

It is planned to extend both this description as well as the wiki pages.

#!/bin/bash
curl -L -o ./src/input/model.tar.gz\
  https://www.kaggle.com/api/v1/models/dfranzen/wb55l_nemomini_fulleval/transformers/default/1/download

mkdir ./src/input/wb55l_nemomini_fulleval/transformers/default/1
tar -xvf ./src/input/model.tar.gz -C ./src/input/wb55l_nemomini_fulleval/transformers/default/1
rm ./src/input/model.tar.gz
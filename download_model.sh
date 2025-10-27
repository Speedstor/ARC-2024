#!/bin/bash
curl -L -o ./input/model.tar.gz\
  https://www.kaggle.com/api/v1/models/dfranzen/wb55l_nemomini_fulleval/transformers/default/1/download

mkdir ./input/wb55l_nemomini_fulleval/transformers/default/1
tar -xvf ./input/model.tar.gz -C ./input/wb55l_nemomini_fulleval/transformers/default/1
rm ./input/model.tar.gz
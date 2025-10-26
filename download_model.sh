#!/bin/bash
curl -L -o ./input/model.tar.gz\
  https://www.kaggle.com/api/v1/models/dfranzen/wb55l_nemomini_fulleval/transformers/default/1/download

tar -xvf ./input/modoel.tar.gz -C ./input/wb55l_nemomini_fulleval
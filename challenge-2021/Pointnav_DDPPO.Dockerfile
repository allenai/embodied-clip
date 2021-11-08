FROM fairembodied/habitat-challenge:testing_2021_habitat_base_docker

RUN /bin/bash -c "rm -rf /habitat-lab"
ADD . /habitat-lab/
RUN /bin/bash -c ". activate habitat; cd /habitat-lab; pip install -r requirements.txt; python setup.py develop --all"

ADD challenge-2021/ddppo_agents.py agent.py
ADD challenge-2021/submission.sh submission.sh
ADD challenge-2021/configs/ configs/
ADD demo.ckpt.pth demo.ckpt.pth

ENV AGENT_CONFIG_FILE "/configs/ddppo_pointnav.yaml"

ENV AGENT_EVALUATION_TYPE remote
ENV TRACK_CONFIG_FILE "/configs/challenge_pointnav2021.local.rgbd.yaml"

CMD ["/bin/bash", "-c", "source activate habitat && export PYTHONPATH=/evalai-remote-evaluation:/habitat-lab:$PYTHONPATH && export CHALLENGE_CONFIG_FILE=$TRACK_CONFIG_FILE && bash submission.sh --model-path demo.ckpt.pth --input-type rgbd"]

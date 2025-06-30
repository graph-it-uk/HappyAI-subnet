<div align="center">

# HappyAI: Bittensor SN103 <!-- omit in toc -->
### Powering the Avocado mental-health companion
</div>

---

## Abstract
Mental-health conditions affect hundreds of millions of people, yet stigma, cost, and limited clinical capacity still keep timely help out of reach. HappyAI set out to change that by building Avocado—an AI companion that offers accessible, empathetic, and evidence-based support around the clock.

Running on our Bittensor subnet, Avocado is available 24 hours a day, 7 days a week, ready whenever users reach for help. Each conversation is guided by proven Cognitive Behavioral Therapy principles, allowing the system to suggest practical exercises and reframing techniques that fit an individual’s situation. With advanced natural-language understanding, Avocado responds in a tone that feels human and supportive, adapting to the user’s emotional state rather than relying on canned replies.

At every step we enforce rigorous safety-escalation protocols. If a user’s messages indicate crisis-level distress, Avocado immediately surfaces local helplines and, where permitted, can hand off to live professionals—ensuring empathy never comes at the expense of responsibility.

This project is just the start. HappyAI will continue creating applications where responsible AI tangibly improves quality of life—whether in wellbeing, productivity, or daily care—always with the same focus on accessibility, personalization, and safety.

---

## Installation

Install requirements:
```
pip install -r requirements.txt
```

Then you can prepare your dotenv file by:
```
cp .env.template .env
```
The OpenAI keys are to be added to .env file


### Start the Miner

First, you need to start the worker, containing the logic of AI assistant:

```
cd worker
uvicorn app.main:app --host 0.0.0.0 --port 1235
cd ..
```

Then you can start the miner by running the following command:
```
python app/neurons/miner.py --netuid N --subtensor.network finney --wallet.name <your_wallet_name> --wallet.hotkey <your_hotkey> --logging.debug --blacklist.force_validator_permit --axon.port 8091
```

(assuming that the current dir is part of python import path. Otherwise add PYTHONPATH=$(pwd) in front of command)

### Start the Validator

You can start the validator by running the following commands:

```
chmod +x run.sh
```
```
./run.sh --netuid N --subtensor.network finney --wallet.name <your_wallet_name> --wallet.hotkey <your_hotkey> --logging.debug
```

Validators are asked to use OpenAI GPT-4o model for evaluation without changes in codebase.

The setup script with autoapdate of code is suggested. Functional changes are to be announced in correspondent Bittensor discord channel.

With no utilisation of fine-tuned self hosted models in the current phase, the subnet is not compute resource heavy. Miner/validator code could be hosted on simple CPU instance as t3.large:
vCPUs: 2
Memory: 8Gb
Storage: 20Gb
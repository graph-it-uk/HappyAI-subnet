<div align="center">

# HappyAI: Bittensor SN103 <!-- omit in toc -->
</div>

---

## Abstract
Mental health issues affect millions globally, influencing individuals’ personal, professional, and social well-being. Nowadays millions face barriers to accessing the care they need. From societal stigma to resource shortages, these challenges leave many struggling alone. Our vision is to harness the power of artificial intelligence to transform mental health care, making it accessible, empathetic, and effective for everyone, anytime and anywhere.

AI has the potential to democratize mental health care by offering immediate, personalized, and evidence-based support. With advanced natural language understanding and therapeutic frameworks like Cognitive Behavioral Therapy (CBT), AI can simulate empathetic, nonjudgmental conversations. It can guide users through reflection, provide actionable advice, and escalate critical issues to ensure safety—all while being available 24/7.

Existing mental health chatbots often lack the sophistication and depth required for truly meaningful support. They struggle to balance empathy with accuracy and to adapt to the nuances of individual emotional states. Improving these systems is essential to building trust and delivering impactful care. By integrating advanced therapeutic techniques, rigorous validation, and a continuous learning model, we strive to elevate the quality and reliability of AI-driven mental health solutions.

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

You can start the validator by running the following command:

```
./run.sh --netuid N --subtensor.network finney --wallet.name <your_wallet_name> --wallet.hotkey <your_hotkey> --logging.debug
```

module.exports = {
      apps : [{
        name   : 'validator_validator_process',
        script : 'python3',
        cwd    : '/home/rataski/HappyAI-subnet',
        interpreter: 'none',
        min_uptime: '5m',
        max_restarts: '5',
        args: ['-m', 'app.neurons.validator', '--netuid','1','--subtensor.network','finney','--wallet.name','crypto_wallet','--wallet.hotkey','validator1','--logging.debug','--blacklist.force_validator_permit','--axon.port','8091'],
        env: {
          PYTHONPATH: '/home/rataski/HappyAI-subnet'
        }
      }]
    }

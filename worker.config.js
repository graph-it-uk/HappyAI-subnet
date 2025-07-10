module.exports = {
      apps : [{
        name   : 'validator_worker_process',
        script : 'python3',
        cwd    : '/home/rataski/HappyAI-subnet/worker',
        interpreter: 'none',
        min_uptime: '5m',
        max_restarts: '5',
        args: ['-m', 'uvicorn', 'app.main:app', '--host', '0.0.0.0', '--port', '1235'],
        env: {
          PYTHONPATH: '/home/rataski/HappyAI-subnet'
        }
      }]
    }

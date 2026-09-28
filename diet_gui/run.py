import subprocess, sys, os
os.chdir(os.path.dirname(__file__))
subprocess.run([sys.executable, 'app.py'])

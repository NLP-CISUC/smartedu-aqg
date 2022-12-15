import logging
import logging.config

logger = logging.getLogger('smartedu-aqg')
ch = logging.FileHandler("app.log")
logger.setLevel(logging.INFO)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
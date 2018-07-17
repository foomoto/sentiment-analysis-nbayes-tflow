import datetime
import enum

from sqlalchemy import *
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
import os

db_username = os.environ['HUNTER_DB_USERNAME']
db_password = os.environ['HUNTER_DB_PASSWORD']
db_host = os.environ['HUNTER_DB_HOST']
db_port = os.environ['HUNTER_DB_PORT']

Base = declarative_base()
engine = create_engine("mysql://%s:%s@%s:%s/octopus_local" % (db_username, db_password, db_host, db_port))
Session = sessionmaker(bind=engine)
session = Session()


class Feature(Base):
    __tablename__ = 'feature'

    id = Column(BigInteger, primary_key=True)
    created = Column(DateTime, default=datetime.datetime.utcnow)
    updated = Column(DateTime, default=datetime.datetime.utcnow)

    key = Column(String(255))
    value = Column(String(255))



Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)
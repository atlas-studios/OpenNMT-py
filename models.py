from unicodedata import name
from sqlalchemy import Integer, String, DateTime, Boolean, text
from sqlalchemy.sql.schema import Column
from database import Base
import datetime

class MT_Model(Base):
    __tablename__ = 'mt_models'

    cbid = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    provider = Column(String, nullable=False)
    created_at = Column(DateTime, server_default=text("now()"))
    last_trained = Column(DateTime, server_default=text("now()"))
    lang_pair = Column(String, nullable=False)
    last_used = Column(DateTime)
    
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    password = Column(String(100), nullable=False)
    email = Column(String(255), nullable=False, unique=True)
    created_at = Column(DateTime, server_default=text("now()"))
    is_admin = Column(Boolean, nullable=False, unique=False, default=False, server_default='f')
    mt_group = Column(String(255))
    charcount = Column(Integer, default=0)
    last_login = Column(DateTime)    

    def set_is_admin(self, is_admin):
        self.is_admin = is_admin

    def print_user(self):
        print(f"Name: {self.name}, Email: {self.email}, Group: {self.mt_group}, Admin: {self.is_admin}, Charcount: {self.charcount}, Last Login: {self.last_login}, Created: {self.created_at}")
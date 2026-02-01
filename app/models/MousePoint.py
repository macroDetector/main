from sqlalchemy import Column, Integer, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import dataclasses


Base = declarative_base()

@dataclasses.dataclass
class MousePoint(Base):
    __tablename__ = "mouse_points"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)
    x = Column(Integer, nullable=False)
    y = Column(Integer, nullable=False)
    deltatime = Column(Float, nullable=False)
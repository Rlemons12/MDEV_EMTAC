import sys
import os
from sqlalchemy.orm import relationship, scoped_session, sessionmaker
from sqlalchemy import (Column, ForeignKey, Integer, String)
if getattr(sys, 'frozen', False):  # Check if running as an executable
    current_dir = os.path.dirname(sys.executable)  # Use the executable directory
else:
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Use the script directory
from config import DATABASE_URL
from base import Base


sys.path.append(current_dir)
class Assembly(Base):
    __tablename__ = 'assembly'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)

    # Relationships
    subassemblies = relationship("SubAssembly", back_populates="assembly")
    
class SubAssembly(Base):
    __tablename__ = 'subassembly'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    assembly_id = Column(Integer, ForeignKey('assembly.id'), nullable=False)
    assembly_view_id = Column(Integer, ForeignKey('assembly_view.id'), nullable=False)

    # Relationships
    assembly = relationship("Assembly", back_populates="subassemblies")
    assembly_view = relationship("AssemblyView", back_populates="subassemblies")

class AssemblyView(Base):
    __tablename__ = 'assembly_view'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)

    # Relationships
    subassemblies = relationship("SubAssembly", back_populates="assembly_view")









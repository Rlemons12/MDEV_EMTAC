from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Tool(Base):
    __tablename__ = 'tool'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(String(255))
    tool_type = Column(String(50))  # Discriminator column

    __mapper_args__ = {
        'polymorphic_identity': 'tool',
        'polymorphic_on': tool_type
    }

    # Relationship to Solution via ToolSolutionAssociation
    tool_solutions = relationship("ToolSolutionAssociation", back_populates="tool")
    solutions = relationship("Solution", secondary='tool_solution', back_populates="tools")

    def __repr__(self):
        return f"<Tool(name='{self.name}', type='{self.tool_type}')>"


class HandTool(Tool):
    __tablename__ = 'hand_tool'

    id = Column(Integer, ForeignKey('tool.id'), primary_key=True)
    handle_material = Column(String(100))  # Specific attribute for HandTool
    is_multi_purpose = Column(Boolean, default=False)  # Indicates if the tool serves multiple functions

    __mapper_args__ = {
        'polymorphic_identity': 'hand_tool',
    }

    def __repr__(self):
        return f"<HandTool(name='{self.name}', handle_material='{self.handle_material}', multi_purpose={self.is_multi_purpose})>"


class PowerTool(Tool):
    __tablename__ = 'power_tool'

    id = Column(Integer, ForeignKey('tool.id'), primary_key=True)
    power_source = Column(String(50))  # e.g., Electric, Battery
    voltage = Column(Float)  # Voltage rating

    __mapper_args__ = {
        'polymorphic_identity': 'power_tool',
    }

    def __repr__(self):
        return (f"<PowerTool(name='{self.name}', power_source='{self.power_source}', "
                f"voltage={self.voltage}V)>")


class MeasuringTool(Tool):
    __tablename__ = 'measuring_tool'

    id = Column(Integer, ForeignKey('tool.id'), primary_key=True)
    measurement_unit = Column(String(20))  # e.g., meters, inches
    accuracy = Column(Float)  # Accuracy in measurement

    __mapper_args__ = {
        'polymorphic_identity': 'measuring_tool',
    }

    def __repr__(self):
        return (f"<MeasuringTool(name='{self.name}', measurement_unit='{self.measurement_unit}', "
                f"accuracy={self.accuracy})>")


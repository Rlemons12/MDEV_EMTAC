class FailureMode:
    def __init__(self, name, severity, occurrence, detection, effect, consequence):
        self.name = name
        self.severity = severity
        self.occurrence = occurrence
        self.detection = detection
        self.effect = effect
        self.consequence = consequence
        self.rpn = self.calculate_rpn()

    def calculate_rpn(self):
        return self.severity * self.occurrence * self.detection

    def update_scores(self, severity=None, occurrence=None, detection=None):
        if severity is not None:
            self.severity = severity
        if occurrence is not None:
            self.occurrence = occurrence
        if detection is not None:
            self.detection = detection
        self.rpn = self.calculate_rpn()

    def __str__(self):
        return (f"Failure Mode: {self.name}, Severity: {self.severity}, "
                f"Occurrence: {self.occurrence}, Detection: {self.detection}, "
                f"Effect: {self.effect}, Consequence: {self.consequence}, RPN: {self.rpn}")


class RCM:
    def __init__(self, system_name, description):
        self.system_name = system_name
        self.description = description
        self.functions = []
        self.functional_failures = []
        self.failure_modes = []
        self.maintenance_strategies = []

    def add_function(self, function_desc):
        self.functions.append(function_desc)
        print(f"Added Function: {function_desc}")

    def add_functional_failure(self, failure_desc):
        self.functional_failures.append(failure_desc)
        print(f"Added Functional Failure: {failure_desc}")

    def add_failure_mode(self, name, severity, occurrence, detection, effect, consequence):
        new_mode = FailureMode(name, severity, occurrence, detection, effect, consequence)
        self.failure_modes.append(new_mode)
        print(f"Added Failure Mode: {new_mode}")

    def add_maintenance_strategy(self, failure_mode_name, strategy_desc):
        self.maintenance_strategies.append({'failure_mode': failure_mode_name, 'strategy': strategy_desc})
        print(f"Added Maintenance Strategy for '{failure_mode_name}': {strategy_desc}")

    def get_high_risk_modes(self, threshold=100):
        return [mode for mode in self.failure_modes if mode.rpn > threshold]

    def generate_report(self):
        print(f"\n--- RCM Report for {self.system_name} ---")
        print(f"System Description: {self.description}\n")

        print("Functions:")
        for func in self.functions:
            print(f" - {func}")

        print("\nFunctional Failures:")
        for failure in self.functional_failures:
            print(f" - {failure}")

        print("\nFailure Modes (FMEA):")
        for mode in self.failure_modes:
            print(f" - {mode}")

        print("\nMaintenance Strategies:")
        for strategy in self.maintenance_strategies:
            print(f" - Failure Mode: {strategy['failure_mode']}, Strategy: {strategy['strategy']}")

        print("\nHigh-Risk Failure Modes (RPN > 100):")
        high_risk = self.get_high_risk_modes()
        if high_risk:
            for mode in high_risk:
                print(f" - {mode}")
        else:
            print("No high-risk failure modes found.")


# Example Usage
if __name__ == "__main__":
    rcm = RCM(system_name="Cooling System", description="System responsible for maintaining safe temperature levels.")

    # Adding Functions
    rcm.add_function("Regulate engine temperature")
    rcm.add_function("Prevent overheating during high loads")

    # Adding Functional Failures
    rcm.add_functional_failure("Inability to maintain optimal temperature")
    rcm.add_functional_failure("Failure to prevent overheating")

    # Adding Failure Modes
    rcm.add_failure_mode("Pump Failure", severity=9, occurrence=3, detection=4,
                         effect="Engine overheating", consequence="System shutdown")
    rcm.add_failure_mode("Coolant Leak", severity=8, occurrence=5, detection=6,
                         effect="Reduced cooling efficiency", consequence="Gradual temperature rise")
    rcm.add_failure_mode("Sensor Malfunction", severity=7, occurrence=4, detection=3,
                         effect="Incorrect temperature readings", consequence="Delayed response to overheating")

    # Adding Maintenance Strategies
    rcm.add_maintenance_strategy("Pump Failure", "Routine pump inspections every 3 months")
    rcm.add_maintenance_strategy("Coolant Leak", "Check coolant levels weekly and replace hoses annually")
    rcm.add_maintenance_strategy("Sensor Malfunction", "Calibrate sensors bi-annually")

    # Generate RCM Report
    rcm.generate_report()

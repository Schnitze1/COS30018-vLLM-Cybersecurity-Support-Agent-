import random

# A list of "normal" system logs
NORMAL_LOGS = [
    "User 'admin' logged in from 192.168.1.5",
    "Service 'sshd' started successfully.",
    "File '/var/www/html/index.html' accessed by user 'webmaster'",
    "User 'guest' logged out.",
    "System backup completed successfully.",
]

# A list of "malicious" or "suspicious" logs
ATTACK_LOGS = [
    "Failed login attempt for user 'root' from 103.22.14.5",
    "SQL injection attempt detected from 192.168.1.10: '... OR 1=1'",
    "Multiple failed login attempts for 'admin' from 203.0.113.8",
    "Suspicious process 'xmr-rig' detected running on server.",
    "File '/etc/passwd' read attempt by unauthorized user 'www-data'",
]


class LogSimulator:
    """
    A simple environment to simulate cybersecurity logs for the agent.
    """
    def __init__(self):
        # We'll use this to decide whether to send an attack or not
        self.attack_probability = 0.3  # 30% chance of the next log being an attack

    def get_next_log(self):
        """
        Returns a new log entry and its ground-truth label.

        Returns:
            (str, int): A tuple containing (log_message, label)
                        where label 0 = normal, 1 = attack
        """
        if random.random() < self.attack_probability:
            # It's an attack
            log_message = random.choice(ATTACK_LOGS)
            label = 1
        else:
            # It's a normal log
            log_message = random.choice(NORMAL_LOGS)
            label = 0

        return log_message, label

# --- This is just for testing the file directly ---
if __name__ == "__main__":
    print("--- Testing Log Simulator ---")
    simulator = LogSimulator()

    for _ in range(10):
        log, log_label = simulator.get_next_log()
        label_text = "ATTACK" if log_label == 1 else "Normal"
        print(f"[{label_text}] {log}")

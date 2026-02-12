# decision.py

class DecisionEngine:

    def decide(self,
               curiosity,
               dyn_thresh,
               act_power,
               conf,
               gate,
               word_freq=0):

        # -------------------------
        # Novel signal
        # -------------------------
        if curiosity > dyn_thresh and word_freq < 3:
            return "investigate"

        # -------------------------
        # Familiar pattern → react faster
        # -------------------------
        if word_freq > 10 and act_power > conf * 0.3:
            return "react"

        # -------------------------
        # Attention gate
        # -------------------------
        if gate > 0.6:
            return "focus"

        # -------------------------
        # Low confidence
        # -------------------------
        if conf < 0.2:
            return "uncertain"

        return "idle"


decision_engine = DecisionEngine()

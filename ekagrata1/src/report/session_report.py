import os
from fpdf import FPDF
import matplotlib.pyplot as plt
from datetime import datetime

class SessionReport:
    def __init__(self, config):
        self.output_dir = "reports"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def _create_graph(self, angle_history, time_history, output_path):
        """
        Generates a matplotlib chart of the elbow angle over time.
        """
        plt.figure(figsize=(8, 4))
        # Ensure we don't try to plot empty lists
        if len(time_history) > 0 and len(angle_history) > 0:
            # Relative time for x-axis
            t0 = time_history[0]
            t_rel = [t - t0 for t in time_history]
            plt.plot(t_rel, angle_history, label="Fused Elbow Angle", color='blue', linewidth=2)
        
        plt.title("Elbow Extension During Session")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Angle (degrees)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def _get_weakest_metric_and_suggestion(self, metrics):
        """
        Finds the lowest scoring component and provides a coaching suggestion.
        """
        if not metrics:
            return "N/A", "No smash data recorded."
            
        scores = {
            'Elbow Extension': metrics['elbow_score'],
            'Wrist Speed': metrics['speed_score'],
            'Trunk Rotation': metrics['trunk_score']
        }
        
        weakest_name = min(scores, key=scores.get)
        
        suggestions = {
            'Elbow Extension': "Try to extend your arm fully just before impact for maximum reach and power.",
            'Wrist Speed': "Focus on snapping your wrist aggressively at the peak of the motion.",
            'Trunk Rotation': "Engage your core and rotate your shoulders into the shot to generate more torque."
        }
        
        return weakest_name, suggestions[weakest_name]

    def generate(self, athlete_name, final_score, metrics, angle_history, time_history):
        """
        Generates a 1-page PDF report.
        """
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_name = f"{self.output_dir}/{athlete_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M')}.pdf"
        
        # 1. Generate Graph image
        graph_img_path = f"{self.output_dir}/temp_graph.png"
        self._create_graph(angle_history, time_history, graph_img_path)
        
        # 2. Get Coaching Feedback
        weakest_joint, suggestion = self._get_weakest_metric_and_suggestion(metrics)
        
        # 3. Build PDF
        pdf = FPDF()
        pdf.add_page()
        
        # Header
        pdf.set_font("Arial", 'B', 24)
        pdf.cell(0, 15, "EKAGRATA Session Report", ln=True, align='C')
        
        pdf.set_font("Arial", 'I', 12)
        pdf.cell(0, 10, f"Athlete: {athlete_name} | Date: {date_str}", ln=True, align='C')
        pdf.ln(10)
        
        # Score Section
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, f"Suggested Technique Score: {final_score}/100", ln=True)
        pdf.set_font("Arial", 'I', 10)
        pdf.cell(0, 8, "Note: This is an AI-suggested score. Validate with a human coach.", ln=True)
        pdf.ln(5)
        
        # Details
        if metrics:
            pdf.set_font("Arial", '', 12)
            pdf.cell(0, 8, f"Max Wrist Speed: {metrics['raw_speed']:.1f} deg/s", ln=True)
            pdf.cell(0, 8, f"Max Elbow Extension: {metrics['raw_elbow']:.1f} deg", ln=True)
            pdf.cell(0, 8, f"Trunk Rotation Angle: {metrics['raw_trunk']:.1f} deg", ln=True)
        pdf.ln(10)
        
        # Graph
        if os.path.exists(graph_img_path):
            pdf.image(graph_img_path, x=15, w=180)
            pdf.ln(10) # Line break after image
            
        # Coaching Suggestion
        pdf.set_y(220) # Move down below the image
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Coaching Insights", ln=True)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, f"Weakest Area: {weakest_joint}", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 8, f"Suggestion: {suggestion}")
        
        # Save
        pdf.output(file_name)
        
        # Cleanup temp image
        if os.path.exists(graph_img_path):
            os.remove(graph_img_path)
            
        print(f"Report generated: {file_name}")
        return file_name

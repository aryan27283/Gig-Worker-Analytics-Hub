import os
from dotenv import load_dotenv
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as Params
from datetime import datetime

load_dotenv()

class GigAI:
    def __init__(self):
        try:
            self.credentials = {
                "apikey": os.getenv("IBM_GRANITE_API_KEY"),
                "url": os.getenv("IBM_GRANITE_URL", "https://us-south.ml.cloud.ibm.com")
            }
            self.project_id = os.getenv("IBM_GRANITE_PROJECT_ID")

            if not all(self.credentials.values()) or not self.project_id:
                raise ValueError("Missing IBM credentials in .env")

            # Model configurations
            self.model = Model(
                model_id="ibm/granite-13b-instruct-v2",
                credentials=self.credentials,
                project_id=self.project_id,
                params={
                    Params.DECODING_METHOD: "greedy",
                    Params.MAX_NEW_TOKENS: 1000,
                    Params.TEMPERATURE: 0.3,
                    Params.REPETITION_PENALTY: 1.2
                }
            )
        except Exception as e:
            raise RuntimeError(f"IBM Granite initialization failed: {str(e)}")

    def generate_judge_ready_recommendations(self, data):
        """Generate impressive, professionally formatted recommendations"""
        try:
            current_date = datetime.now().strftime("%B %d, %Y")
            prompt = f"""As a gig economy expert, create a comprehensive performance report that would impress judges in a competition. 

            Current Date: {current_date}
            Data Sample:
            {data.head().to_string()}

            Format your response as a formal business report with these sections:

            # Gig Work Performance Analysis Report

            ## Executive Summary
            - Overall performance rating (1-5 stars)
            - Key achievements
            - Growth potential

            ## Performance Metrics
            - Hourly earnings: ${data['earnings'].sum()/data['hours'].sum():.2f}
            - Efficiency metrics
            - Platform comparisons

            ## Optimal Strategy Analysis
            ### Best Performing Days
            - Detailed day-by-day analysis
            - Expected earnings potential

            ### Peak Time Windows
            - Hourly breakdown of profitability
            - Platform-specific recommendations

            ## Action Plan
            1. Primary recommendation with ROI estimate
            2. Secondary recommendation
            3. Risk mitigation strategies

            ## Future Projections
            - 30-day earnings forecast
            - Growth opportunities
            - Sustainability considerations

            Use professional business language with data-driven insights. Include specific numbers and percentages where possible."""
            
            return self.model.generate_text(prompt).strip()
        except Exception as e:
            return f"Error generating report: {str(e)}"

    def answer_question(self, question):
        """Generate expert responses with academic rigor"""
        try:
            prompt = f"""As a gig economy professor preparing competition-level analysis, answer with academic rigor:

            Question: {question}

            Structure your response with:

            ### Thesis Statement
            [Core argument]

            ### Supporting Data
            - Statistic 1 with source
            - Statistic 2 with source

            ### Case Studies
            1. Relevant example 1
            2. Relevant example 2

            ### Implementation Strategy
            - Step-by-step action plan
            - Expected outcomes

            ### References
            - Academic sources
            - Industry reports"""
            
            return self.model.generate_text(prompt).strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"
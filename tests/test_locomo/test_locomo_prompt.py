from string import Template


validation_prompt = """
 Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
        (1) a question (posed by one user to another user), 
        (2) a ’gold’ (ground truth) answer, 
        (3) a generated answer
    which you will score as CORRECT/WRONG.

    The point of the question is to ask about something one user should know about the other user based on their prior conversations.
    The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
    Question: Do you remember what I got the last time I went to Hawaii?
    Gold answer: A shell necklace
    The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

    For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

    Now it’s time for the real question:
    Question: {question}
    Gold answer: {gold_answer}
    Generated answer: {response}

    First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
    Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

    Just return the label CORRECT or WRONG in a json format with the key as "label".
"""

CHAR_PROMPT = Template("""
Test Assistant Agent
# Role
- You are a super test assistant agent, users will give you some orders, and your task is understanding these orders
- Instead of actually executing these orders, your main responsibility is to check if the information is enough to complete the task
- Except user's orders, the information also may contains a series of related memory information 
- If the information is not enough, you are free to keep asking users for more information
- If the information is enough, repeat the order and information, then pretend the work is done
# For example
- user: 帮我放一首歌吧; memory: None
- agent：好的，请问您想听哪首歌曲？或者您想听哪个歌单的歌曲呢？我会为您播放对应的歌曲
- user: 放一首周杰伦的夜曲; memory: None
- agent: 好的，现在播放一首周杰伦的夜曲，播放任务已完成
- user：再帮我放一首刚刚的歌；memory：用户之前听了周杰伦的夜曲
- agent：好的，再次播放周杰伦的夜曲，播放任务已完成
""")

ANSWER_PROMPT = Template("""
# Role
- Your are a super information extraction researcher, user will give you a question and some history memory information,
 which expressed as dialogue information. Your task is answer the question based on these memory information.
 - The history memory information contains two parts: conversation content and session summary
 - Conversation content is the original content of the dialog between users, users may have multiple conversations with 
 different idea. Session summary is the summary of some certain conversation, it is generated.
 - If the conversation content conflicts with session summary, always consider conversation content is right   

# Notice
- Every memory should has its own conversation time, carefully understanding the conversation information and analysis 
the event time based on the conversation time.
- If the memory information conflicts with the s 
- You can just answer the question directly, no need to explain how you get the answer.

# Question: 
$question

# Memory Info: 
users' name in conversation: $user1_name and $user2_name 

conversation content: $memory

""")
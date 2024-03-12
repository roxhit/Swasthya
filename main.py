import streamlit as st
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from tools import ImageCaptionTool, ObjectDetectionTool
import tempfile
from gtts import gTTS  # Text-to-Speech library
import speech_recognition as sr  # Library for speech recognition

# Initialize history list
history = []

# Specify the tools
tools = [ImageCaptionTool(), ObjectDetectionTool()]

# Specify the conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

# Initialize the ChatOpenAI agent
llm = ChatOpenAI(
    openai_api_key='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',
    temperature=0,
    model_name='gpt-3.5-turbo'
)

agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    max_iterations=50,
    verbose=True,
    memory=conversational_memory,
    early_stoppy_method='generate'
)

# Set up Streamlit UI
st.title("Swasthya")

option = st.radio("Select option", ("Upload Image", "Voice Command"))

# Check the selected option
if option == "Upload Image":
    st.subheader("Please Upload An Image")
    # File uploader
    file = st.file_uploader("", type=["jpeg", "jpg", "png"])
    
    if file:
        st.image(file, use_column_width=True)
        user_question = st.text_input('Ask a Question')
        
        # Create a temporary file in the system's default temporary directory
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(file.getbuffer())
            image_path = f.name

            # Check if the user question is provided
            if user_question and user_question != "":
                # Run the agent with the user question and image path
                response = agent.run('{},this is a image path:{}'.format(user_question, image_path))

                # Add search query to history
                history.append((user_question, response))

                # Generate speech from response
                tts = gTTS(text=response, lang='en')
                with tempfile.NamedTemporaryFile(delete=False) as f:
                    tts_path = f.name
                    tts.save(tts_path)

                # Play the audio
                st.audio(tts_path, format='audio/mp3', start_time=0)

                # Display the response
                st.write(response)
elif option == "Voice Command":
    st.subheader("Voice Command")

    # Create a function to capture voice command
    def capture_voice_command():
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("Listening...")
            audio = r.listen(source)
        
        try:
            st.write("Processing...")
            command = r.recognize_google(audio)
            return command
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand your command.")
            return None
        except sr.RequestError as e:
            st.error(f"Could not request results; {e}")
            return None

    # Capture voice command
    user_command = st.button("Start Recording")

    if user_command:
        command = capture_voice_command()
        if command:
            st.write("You said:", command)
            # Process the user command
            response = agent.run(command)

            # Add search query to history
            history.append((command, response))

            # Generate speech from response
            tts = gTTS(text=response, lang='en')
            with tempfile.NamedTemporaryFile(delete=False) as f:
                tts_path = f.name
                tts.save(tts_path)

            # Play the audio
            st.audio(tts_path, format='audio/mp3', start_time=0)

            # Display the response
            st.write("Response:", response)

# Show search history
st.sidebar.header("Search History")
for i, (query, response) in enumerate(history[::-1], start=1):
    st.sidebar.subheader(f"Query {i}: {query}")
    st.sidebar.write(f"Response: {response}")

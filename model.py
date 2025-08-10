# Enhanced AI Agent with Google Search & Calculator - Fully Debugged
# Requirements: pip install langchain langchain-openai langchain-community python-dotenv requests psutil webbrowser googlesearch-python duckduckgo-search

import math
import subprocess
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from urllib.parse import quote_plus

import psutil
from dotenv import load_dotenv
# Updated imports
from langchain.tools import Tool
from langchain_community.llms import Ollama
from langchain_community.utilities import WikipediaAPIWrapper


from Chatbot import ChatBot

# Try to import OpenAI
try:
    from langchain_openai import OpenAI, ChatOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Try to import Google search (via duckduckgo_search)
try:
    from duckduckgo_search import DDGS

    SEARCH_AVAILABLE = True
except ImportError:
    SEARCH_AVAILABLE = False

load_dotenv()


@dataclass
class AgentResponse:
    reasoning: str
    plan: List[str]
    action: str
    result: str
    confidence: float = 0.0


class Calculator:
    """Safe calculator for mathematical expressions"""

    @staticmethod
    def evaluate(expression: str) -> str:
        """Safely evaluate mathematical expressions"""
        try:
            # Clean the expression
            expression = expression.strip()

            # Remove common prefixes
            prefixes = ['calc', 'calculate', 'calculator', 'cal', '=']
            for prefix in prefixes:
                if expression.lower().startswith(prefix):
                    expression = expression[len(prefix):].strip()

            # Only allow safe characters
            allowed_chars = set('0123456789+-*/().^ ')
            if not all(c in allowed_chars for c in expression):
                return f"❌ Invalid characters in expression: {expression}"

            # Handle power operations
            expression = expression.replace('^', '**')

            # Evaluate safely using eval with restricted globals
            safe_dict = {
                "__builtins__": {},
                "abs": abs,
                "round": round,
                "max": max,
                "min": min,
                "pow": pow,
                "sqrt": math.sqrt,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "pi": math.pi,
                "e": math.e
            }

            result = eval(expression, safe_dict, {})

            return f"🔢 {expression} = {result}"

        except ZeroDivisionError:
            return f"❌ Division by zero error in: {expression}"
        except Exception as e:
            return f"❌ Math error: {str(e)} in expression: {expression}"


class GoogleSearchTool:
    """Search tool with rich snippets, powered by DuckDuckGo"""

    def __init__(self):
        self.search_available = SEARCH_AVAILABLE

    def search(self, query: str, max_results: int = 5) -> str:
        """Perform a web search and return rich results with snippets and links."""
        if not self.search_available:
            return "❌ No search engines available. Install: pip install duckduckgo-search"

        try:
            from RealtimeSearchEngine import RealtimeSearchEngine

            return f"🔍 Search results for '{query}':\n" + f"\n{RealtimeSearchEngine(query)}"

        except Exception as e:
            return f"❌ Search error: {str(e)}"


class EnhancedLocalLLM:
    """Enhanced local LLM with expanded knowledge and better pattern matching"""

    def __call__(self, prompt: str) -> str:
        """Enhanced response generation with better pattern matching"""
        prompt_lower = prompt.lower().strip()
        original_prompt = prompt.strip()

        # Check for model queries
        if any(phrase in prompt_lower for phrase in ['which model', 'what model', 'current model']):
            return "I'm currently using the Enhanced Local LLM with expanded knowledge base."

        # Rest of the existing code...
    def __init__(self, temperature: float = 0.7):
        self.temperature = temperature
        self.conversation_history = []

        # Expanded knowledge base
        self.knowledge_base = {
            'quantum computing': """Quantum computing is a revolutionary computing paradigm that leverages quantum mechanical phenomena like superposition and entanglement to process information. Unlike classical computers that use bits (0 or 1), quantum computers use quantum bits or 'qubits' that can exist in multiple states simultaneously.

Key concepts:
- Superposition: Qubits can be in multiple states at once
- Entanglement: Qubits can be correlated across space
- Quantum gates: Operations that manipulate qubits
- Applications: Cryptography, optimization, drug discovery, financial modeling

Current leaders: IBM, Google, Microsoft, IonQ, Rigetti""",

            'artificial intelligence': """Artificial Intelligence (AI) refers to computer systems that can perform tasks typically requiring human intelligence. This includes learning, reasoning, perception, and decision-making.

Types of AI:
- Narrow AI: Specialized for specific tasks (current state)
- General AI: Human-level intelligence across domains (future goal)
- Machine Learning: Learning from data without explicit programming
- Deep Learning: Neural networks with multiple layers
- Natural Language Processing: Understanding and generating human language

Applications: Healthcare, finance, transportation, entertainment, robotics""",

            'machine learning': """Machine Learning (ML) is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.

Main types:
- Supervised Learning: Learning with labeled examples
- Unsupervised Learning: Finding patterns in unlabeled data
- Reinforcement Learning: Learning through trial and error with rewards

Common algorithms:
- Linear/Logistic Regression
- Decision Trees and Random Forests
- Support Vector Machines
- Neural Networks
- Clustering (K-means, hierarchical)""",

            'python programming': """Python is a high-level, interpreted programming language known for its simplicity and readability.

Key features:
- Easy to learn and read
- Extensive standard library
- Large ecosystem of packages (pip)
- Cross-platform compatibility
- Used for: web development, data science, AI/ML, automation, scripting

Basic syntax:
- Variables: x = 5
- Functions: def my_function():
- Lists: my_list = [1, 2, 3]
- Loops: for item in list:
- Conditions: if condition:""",

            'blockchain': """Blockchain is a distributed ledger technology that maintains a continuously growing list of records (blocks) linked and secured using cryptography.

Key features:
- Decentralized: No single point of control
- Immutable: Records cannot be altered
- Transparent: All transactions visible
- Consensus: Agreement mechanisms for validation

Applications:
- Cryptocurrencies (Bitcoin, Ethereum)
- Smart contracts
- Supply chain tracking
- Digital identity
- Voting systems"""
        }


    def __call__(self, prompt: str) -> str:
        """Enhanced response generation with better pattern matching"""
        prompt_lower = prompt.lower().strip()
        original_prompt = prompt.strip()

        # Store conversation
        self.conversation_history.append({"role": "user", "content": prompt})

        # Check for math expressions first
        if any(char in prompt for char in '+-*/=()') and any(char.isdigit() for char in prompt):
            if any(word in prompt_lower for word in ['calc', 'calculate', 'cal', '+']):
                calc = Calculator()
                response = calc.evaluate(original_prompt)
                self.conversation_history.append({"role": "assistant", "content": response})
                return response

        # Check knowledge base
        for key, value in self.knowledge_base.items():
            if key.replace(' ', '') in prompt_lower.replace(' ', '') or any(
                    word in prompt_lower for word in key.split()):
                response = value
                self.conversation_history.append({"role": "assistant", "content": response})
                return response

        # Check for code requests
        if any(word in prompt_lower for word in ['code', 'program', 'write', 'python', 'function', 'loop']):
            if 'hello world' in prompt_lower:
                response = self.code_templates['python hello world']
            elif 'function' in prompt_lower:
                response = self.code_templates['python function']
            elif 'loop' in prompt_lower:
                response = self.code_templates['python loop']
            elif 'python' in prompt_lower:
                response = """I can help with Python programming! Here are some examples:

• Basic syntax: variables, functions, loops
• Data structures: lists, dictionaries, sets
• File operations: reading/writing files
• Web scraping and APIs
• Data analysis with pandas
• Machine learning with scikit-learn

What specific Python topic would you like help with?"""
            else:
                response = f"I can help with coding! What programming language or specific task are you working on? I have examples for Python, JavaScript, and more."

        # Pattern matching for common requests
        elif any(word in prompt_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            response = "Hello! I'm your enhanced AI assistant. I can help with automation, calculations, search, file operations, and answer questions about technology, science, and programming."

        elif 'weather' in prompt_lower:
            response = "I'll search for current weather information using the web search tool."

        elif any(word in prompt_lower for word in ['time', 'clock']):
            response = f"🕐 The current time is {datetime.now().strftime('%H:%M:%S')} on {datetime.now().strftime('%Y-%m-%d %A')}."

        elif any(word in prompt_lower for word in ['date', 'today']):
            response = f"📅 Today's date is {datetime.now().strftime('%Y-%m-%d %A')}."

        elif any(word in prompt_lower for word in ['price', 'cost', 'stock', 'bitcoin', 'gold']):
            response = "I'll search for current pricing information using Google search."

        elif 'system' in prompt_lower and any(word in prompt_lower for word in ['info', 'status', 'check']):
            response = "Let me check your system information."

        elif any(word in prompt_lower for word in ['thank', 'thanks']):
            response = "You're welcome! I'm here to help with any other questions or tasks you might have."

        else:
            # Generic helpful response
            response = f"I understand you're asking about: '{original_prompt}'. I can help with:\n\n• 🔍 Web search and current information\n• 🧮 Mathematical calculations\n• 💻 Programming and coding help\n• 📊 Technology explanations\n• ⚙️ System automation\n\nWould you like me to search for more information or help with something specific?"

        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def predict(self, text: str) -> str:
        """Compatibility method for LangChain"""
        return self(text)


class FileProcessor:
    """Handle file reading and processing operations"""

    @staticmethod
    def read_text_file(filepath: str) -> str:
        """Read and return content of text file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                return content if content.strip() else "File is empty"
        except FileNotFoundError:
            return f"File not found: {filepath}"
        except PermissionError:
            return f"Permission denied: {filepath}"
        except Exception as e:
            return f"Error reading file: {str(e)}"

    @staticmethod
    def process_pdf(filepath: str) -> str:
        """Extract text from PDF"""
        try:
            import PyPDF2
            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text if text.strip() else "PDF appears to be empty or contains only images"
        except ImportError:
            return "PDF processing requires PyPDF2: pip install PyPDF2"
        except Exception as e:
            return f"Error processing PDF: {str(e)}"

    @staticmethod
    def summarize_content(content: str, max_length: int = 500) -> str:
        """Create a summary of content"""
        if len(content) <= max_length:
            return content

        sentences = content.split('. ')
        if len(sentences) <= 3:
            return content

        summary = sentences[0] + '. '
        middle_idx = len(sentences) // 2
        summary += sentences[middle_idx] + '. '
        summary += sentences[-1]

        return summary if len(summary) <= max_length else summary[:max_length] + "..."

def open_application(app: str) -> str:
    """Open application by name with cross-platform support"""
    try:
        from AppOpener import close, open as appopen
        # Special cases for websites
        websites = {
            'facebook': 'https://facebook.com',
            'youtube': 'https://youtube.com',
            'instagram': 'https://instagram.com',
            'twitter': 'https://twitter.com',
            'telegram': 'https://web.telegram.org',
            'chrome': 'https://google.com',
            'google': 'https://google.com',
            'whatsapp': 'https://web.whatsapp.com'
        }

        if app.lower() in websites:
            webbrowser.open(websites[app.lower()])
            return f"Opened {app} in browser"

        # Use AppOpener for installed applications
        appopen(app, match_closest=True, output=False, throw_error=True)
        return f'{app} is opened '
    except Exception as e:
        error_msg = f"App '{app}' not installed or error opening"
        print(f"[red]{error_msg}: {e}[/red]")
        return error_msg

class SystemController:
    """Handle system-level operations and app control"""

    @staticmethod
    def open_application(app: str) -> str:
        """Open application by name with cross-platform support"""
        try:
            from AppOpener import close, open as appopen
            # Special cases for websites
            websites = {
                'facebook': 'https://facebook.com',
                'youtube': 'https://youtube.com',
                'instagram': 'https://instagram.com',
                'twitter': 'https://twitter.com',
                'telegram': 'https://web.telegram.org',
                'chrome': 'https://google.com',
                'google': 'https://google.com',
                'whatsapp': 'https://web.whatsapp.com'
            }

            if app.lower() in websites:
                webbrowser.open(websites[app.lower()])
                return f"Opened {app} in browser"

            # Use AppOpener for installed applications
            appopen(app, match_closest=True, output=False, throw_error=True)
            return f'{app} is opened '
        except Exception as e:
            error_msg = f"App '{app}' not installed or error opening"
            print(f"[red]{error_msg}: {e}[/red]")
            return error_msg

    @staticmethod
    def close_application(app_name: str) -> str:
        """Close application by name"""
        try:
            app_name = app_name.strip().lower()
            found = False

            for proc in psutil.process_iter(['pid', 'name']):
                proc_name = proc.info['name'].lower()
                if app_name in proc_name or proc_name.startswith(app_name):
                    proc.terminate()
                    found = True

            if found:
                return f"✅ Successfully closed {app_name}"
            else:
                return f"⚠️  Application {app_name} not found running"

        except Exception as e:
            return f"❌ Error closing {app_name}: {str(e)}"

    @staticmethod
    def get_system_info() -> str:
        """Get comprehensive system information"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            boot_time = datetime.fromtimestamp(psutil.boot_time())
            uptime = datetime.now() - boot_time

            info = f"""💻 System Information:
🔹 CPU Usage: {cpu_percent}%
🔹 Memory: {memory.percent}% used ({memory.used // (1024 ** 3):.1f}GB / {memory.total // (1024 ** 3):.1f}GB)
🔹 Disk: {disk.percent}% used ({disk.used // (1024 ** 3):.1f}GB / {disk.total // (1024 ** 3):.1f}GB)
🔹 Uptime: {str(uptime).split('.')[0]}
🔹 Boot Time: {boot_time.strftime('%Y-%m-%d %H:%M:%S')}"""
            return info
        except Exception as e:
            return f"❌ Error getting system info: {str(e)}"


class MediaController:
    """Enhanced media playback operations with YouTube support"""

    @staticmethod
    def play_music_youtube(query: str = "") -> str:
        """Play music on YouTube"""
        try:
            if not query or query.strip() == "":
                webbrowser.open("https://music.youtube.com")
                return "🎵 Opened YouTube Music"
            else:
                search_query = quote_plus(query.strip())
                youtube_search_url = f"https://www.youtube.com/results?search_query={search_query}"
                webbrowser.open(youtube_search_url)
                return f"🎵 Searching YouTube for: '{query}'"

        except Exception as e:
            return f"❌ Error playing music: {str(e)}"

    @staticmethod
    def play_music_spotify(query: str = "") -> str:
        """Play music on Spotify"""
        try:
            if not query or query.strip() == "":
                webbrowser.open("https://open.spotify.com")
                return "🎵 Opened Spotify"
            else:
                search_query = quote_plus(query.strip())
                spotify_search_url = f"https://open.spotify.com/search/{search_query}"
                webbrowser.open(spotify_search_url)
                return f"🎵 Searching Spotify for: '{query}'"
        except Exception as e:
            return f"❌ Error opening Spotify: {str(e)}"


class AIAgent:
    """Enhanced AI Agent with Google search and calculator"""

    def __init__(self, llm_type: str = "auto"):
        # Initialize components
        self.file_processor = FileProcessor()
        self.system_controller = SystemController()
        self.media_controller = MediaController()

        self.search_tool = GoogleSearchTool()
        self.calculator = Calculator()

        # Track model information
        self.model_info = "Unknown"

        # Initialize LLM
        self.llm = self._initialize_llm(llm_type)

        # Rest of the initialization code...
        # Initialize Wikipedia
        try:
            self.wikipedia = WikipediaAPIWrapper()
            self.wikipedia_available = True
        except Exception:
            self.wikipedia_available = False

        # Define tools
        self.tools = self._create_tools()

        # Create enhanced executor
        self.agent_executor = self._create_enhanced_executor()

    def _initialize_llm(self, llm_type: str):
        """Initialize the appropriate LLM"""
        if llm_type == "auto":
            llm_type = self._detect_best_llm()

        if llm_type == "openai" and OPENAI_AVAILABLE:
            api_key = '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'
            if api_key:
                try:
                    llm = ChatOpenAI(temperature=0.7, openai_api_key=api_key, model="gpt-5")

                    self.model_info = "OpenAI GPT-3.5-turbo"
                    print("✅ OpenAI API initialized successfully")
                    return llm
                except Exception as e:
                    print(f"⚠️  OpenAI API failed: {e}")
                    print("Falling back to local LLM...")

        elif llm_type == "ollama":
            try:
                llm = Ollama(model="llama2")
                self.model_info = "Ollama (Llama2)"
                print("✅ Ollama initialized successfully")
                return llm
            except Exception as e:
                print(f"⚠️  Ollama failed: {e}")
                print("Falling back to local LLM...")

        self.model_info = "Enhanced Local LLM with expanded knowledge base"
        print("✅ Using enhanced local LLM with expanded knowledge base")
        return EnhancedLocalLLM()

    def _detect_best_llm(self) -> str:
        """Detect the best available LLM"""
        api_key = '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'
        if api_key and OPENAI_AVAILABLE:
            return "openai"

        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return "ollama"
        except:
            pass

        return "local"

    def _create_tools(self) -> List[Tool]:
        """Create enhanced tools for the agent - FIXED"""
        tools = [
            Tool(
                name="calculator",
                description="Perform mathematical calculations. Input math expressions like '2+2', '58+54-8*5', '867-87-9/7'",
                func=self.calculator.evaluate
            ),
            Tool(
                name="google_search",
                description="Search Google for current information, prices, news, etc. Provide search query as input",
                func=self.search_tool.search
            ),
            Tool(
                name="system_info",
                description="Get system information including CPU, memory, disk usage",
                func=lambda x: self.system_controller.get_system_info()
            ),
            Tool(
                name="open_app",
                description="Open applications like calculator, notepad, browser. Provide app name as input",
                func=self.system_controller.open_application
            ),
            Tool(
                name="close_app",
                description="Close running applications. Provide app name as input",
                func=self.system_controller.close_application
            ),
            Tool(
                name="play_music_youtube",
                description="Play music. Provide song/artist name as input",
                func=self.media_controller.play_music_youtube
            ),
            Tool(
                name="play_music_spotify",
                description="Play music on Spotify. Provide song/artist name as input",
                func=self.media_controller.play_music_spotify
            ),
            Tool(
                name="read_file",
                description="Read and process files. Provide the file path as input",
                func=self._handle_file_reading
            )
        ]

        if self.wikipedia_available:
            tools.append(
                Tool(
                    name="wikipedia_search",
                    description="Search Wikipedia for factual information. Provide topic as input",
                    func=self._safe_wikipedia_search
                )
            )

        return tools

    def _safe_wikipedia_search(self, query: str) -> str:
        """Safe Wikipedia search with error handling"""
        try:
            result = self.wikipedia.run(query)
            return f"📚 Wikipedia: {result}"
        except Exception as e:
            return f"❌ Wikipedia search unavailable: {str(e)}"

    def _create_enhanced_executor(self):
        """Create enhanced executor with better command routing"""

        class EnhancedExecutor:
            def __init__(self, tools, llm, agent_instance):
                self.tools = {tool.name: tool for tool in tools}
                self.llm = llm
                self.agent = agent_instance

            def run(self, input_text: str) -> str:
                """Enhanced execution logic with better pattern matching and routing"""
                input_lower = input_text.lower().strip()
                original_input = input_text.strip()

                # --- START of Improved Routing Logic ---

                # Application control (high priority)

                if 'open' in input_lower and self._contains_app_name(input_lower):
                    app_name = open_application(input_text[5:])
                    return self.tools['open_app'].func(app_name)
                if 'close' in input_lower and self._contains_app_name(input_lower):
                    app_name = self._extract_app_name(input_text, 'close')
                    return self.tools['close_app'].func(app_name)

                # System commands
                if any(phrase in input_lower for phrase in ['system info', 'system status', 'check system']):
                    return self.tools['system_info'].func("")

                # Calculator (check after specific word-based commands)
                if self._is_math_expression(original_input):
                    return self.tools['calculator'].func(original_input)

                # Music commands
                if 'play' in input_lower and (
                        'music' in input_lower or 'song' in input_lower or '' in input_lower or 'on spotify' in input_lower):
                    music_query = self._extract_music_query(input_text)
                    if 'spotify' in input_lower:
                        return self.tools['play_music_spotify'].func(music_query)
                    else:  # Default to YouTube
                        return self.tools['play_music_youtube'].func(music_query)

                # File operations
                if any(phrase in input_lower for phrase in ['read file', 'open file', 'show file']):
                    filename = self._extract_filename(input_text)
                    if filename:
                        return self.tools['read_file'].func(filename)
                    else:
                        return "Please specify a filename to read (e.g., 'read file.txt')"

                # Search commands (prioritize for current events/prices)
                if any(phrase in input_lower for phrase in
                       ['search for', 'search', 'price of', 'cost of', 'what is the price']):
                    search_query = original_input.replace('search for', '').strip()
                    return self.tools['google_search'].func(search_query)

                # Wikipedia for factual "what is" queries
                if any(phrase in input_lower for phrase in
                       [ 'wiki','wikipedia']) and self.agent.wikipedia_available:
                    wiki_query = original_input.replace('what is', '').replace('who is', '').replace('tell me about',
                                                                                                     '').strip()
                    return self.tools['wikipedia_search'].func(wiki_query)

                # --- END of Improved Routing Logic ---

                # Fallback to LLM for general conversation
                try:
                    return ChatBot(original_input)
                except Exception as e:
                    # Provide a helpful error message if LLM fails
                    return f"I understand you're asking about '{original_input}', but I encountered an issue. Available commands:\n• 🧮 Calculator: '2+2'\n• 🔍 Search: 'search for [topic]'\n• 💻 System: 'open calculator'"

            def _is_math_expression(self, text: str) -> bool:
                """Check if text appears to be a mathematical expression"""
                text = text.strip()
                # If it starts with a calc keyword, it's a math expression.
                if any(text.lower().startswith(keyword) for keyword in ['calc', 'calculate', 'cal', '=']):
                    return True

                # If it contains numbers and operators, it's likely a math expression.
                has_numbers = any(char.isdigit() for char in text)
                has_operators = any(op in text for op in ['+', '-', '*', '/', '^'])

                # Avoid classifying things like "search for python 3.9" as math
                has_letters = any(char.isalpha() for char in text)
                if has_letters and not any(text.lower().startswith(keyword) for keyword in ['calc', 'calculate']):
                    return False

                return has_numbers and has_operators

            def _contains_app_name(self, text: str) -> bool:
                """Check if text contains application names"""
                app_names = ['app', 'application', 'calculator', 'calc', 'notepad', 'browser', 'chrome', 'firefox',
                             'word', 'excel', 'explorer', 'cmd', 'terminal', 'paint']
                return any(app in text for app in app_names)

            def _extract_app_name(self, text: str, command: str) -> str:
                """Extract application name from command"""
                text_lower = text.lower()
                # Remove the command word to find the app name
                query = text_lower.replace(command, '').strip()

                # Check for common app names in the remaining text
                common_apps = ['calculator', 'calc', 'notepad', 'browser', 'chrome', 'firefox', 'word', 'excel',
                               'explorer', 'cmd', 'terminal', 'paint']
                for app in common_apps:
                    if app in query:
                        return app

                # If no specific app found, return the first word after the command
                return query.split()[0] if query.split() else 'calculator'

            def _extract_music_query(self, text: str) -> str:
                """Extract music query from command"""
                query = text.lower()
                remove_words = ['play', 'music', 'song', 'on', 'youtube', 'spotify', 'from']

                for word in remove_words:
                    query = query.replace(word, ' ')

                return query.strip()

            def _extract_filename(self, text: str) -> Optional[str]:
                """Extract filename from text"""
                words = text.split()
                for word in words:
                    if '.' in word and len(word) > 3:  # Likely a filename
                        return word
                return None

        return EnhancedExecutor(self.tools, self.llm, self)


    def _handle_file_reading(self, filepath: str) -> str:
        """Handle file reading and processing"""
        if filepath.lower().endswith('.pdf'):
            content = self.file_processor.process_pdf(filepath)
        else:
            content = self.file_processor.read_text_file(filepath)

        # Summarize if content is very long
        if len(content) > 1000 and not content.startswith("Error") and not content.startswith("File not found"):
            summary = self.file_processor.summarize_content(content)
            return f"📄 File content summary:\n{summary}\n\n💡 [Full content available if needed - file has {len(content)} characters]"
        else:
            return f"📄 File content:\n{content}"

    def process_request(self, user_input: str) -> AgentResponse:
        """Process user requests with enhanced analysis"""
        try:
            result = self.agent_executor.run(user_input)
            action = "Executed via enhanced routing system"
            confidence = 0.95
            reasoning = "Request routed to the appropriate tool."
            plan = ["Analyze request", "Route to tool", "Execute"]

        except Exception as e:
            result = f"❌ Error: {str(e)}"
            action = "Error during execution"
            confidence = 0.3
            reasoning = f"An unexpected error occurred: {e}"
            plan = ["Attempt execution", "Catch exception"]

        return AgentResponse(
            reasoning=reasoning,
            plan=plan,
            action=action,
            result=result,
            confidence=confidence
        )

    def chat(self, user_input: str, show_internal_process: bool = False) -> str:
        """Enhanced chat interface"""
        response = self.process_request(user_input)

        if show_internal_process:
            formatted_response = f"""
🧠 Reasoning: {response.reasoning}

📋 Plan:
{chr(10).join(f"   {i + 1}. {step}" for i, step in enumerate(response.plan))}

⚡ Action: {response.action}

✅ Result: {response.result}

🎯 Confidence: {response.confidence:.1%}
            """
        else:
            formatted_response = response.result

        return formatted_response.strip()


def main():
    """Enhanced main function with better user experience"""
    try:
        print("🚀 Initializing Enhanced AI Agent with Google Search & Calculator...")
        print("   📦 Loading components...")

        # Check for required packages
        missing_packages = []
        if not SEARCH_AVAILABLE:
            missing_packages.append("duckduckgo-search")

        if missing_packages:
            print(f"⚠️  Optional packages missing: {', '.join(missing_packages)}")
            print("   For best experience, install: pip install duckduckgo-search")

        # Initialize the agent
        agent = AIAgent(llm_type="auto")

        print("🤖 Enhanced AI Agent initialized successfully!")
        print(f"   🔍 Search: {'DuckDuckGo' if SEARCH_AVAILABLE else 'None'}")
        print("   🧮 Calculator: Built-in")
        print("   🧠 LLM: Enhanced Local with expanded knowledge")

        print("\n🎯 Available Commands:")
        print("┌─ 💬 General Conversation & Knowledge")
        print("│  • 'What is quantum computing?'")
        print("│  • 'Explain machine learning'")
        print("│  • 'Write Python hello world'")
        print("│")
        print("├─ 🧮 Calculator")
        print("│  • '58+54-8*5' or 'calc 867-87-9/7'")
        print("│  • 'calculate 2^8 + sqrt(16)'")
        print("│")
        print("├─ 🔍 Google Search")
        print("│  • 'What's the price of gold today?'")
        print("│  • 'Search for GTA 6 release date'")
        print("│  • 'Current Bitcoin price'")
        print("│")
        print("├─ 🎵 Music & Media")
        print("│  • 'Play Bohemian Rhapsody on YouTube'")
        print("│  • 'Play jazz on Spotify'")
        print("│")
        print("├─ 💻 System Control")
        print("│  • 'Open calculator/notepad/browser'")
        print("│  • 'Close [app name]'")
        print("│  • 'Check system status'")
        print("│")
        print("├─ ⏰ Reminders")
        print("│  • 'Remind me to call mom|in 1 hour'")
        print("│  • 'Remind me to meeting|in 30 minutes'")
        print("│  • 'List reminders'")
        print("│")
        print("├─ 📄 File Operations")
        print("│  • 'Read file.txt'")
        print("│  • 'Open document.pdf'")
        print("│")
        print("└─ 🛠️ Special Commands")
        print("   • 'debug on/off' - Toggle detailed process display")
        print("   • 'help' - Show command reference")
        print("   • 'quit/exit' - Exit the application")

        print("\n" + "=" * 70)

        debug_mode = False

        while True:
            try:
                user_input = input("\n👤 You: ").strip()

                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print("👋 Thanks for using the Enhanced AI Agent! Goodbye!")
                    break

                if user_input.lower() == 'debug on':
                    debug_mode = True
                    print("🔧 Debug mode enabled - You'll see detailed process information")
                    continue
                elif user_input.lower() == 'debug off':
                    debug_mode = False
                    print("🔧 Debug mode disabled - Showing only results")
                    continue
                elif user_input.lower() in ['help', 'commands']:
                    print("\n📖 Quick Command Examples:")
                    print("🧮 Calculator: '2+2*3', 'calc 100/5'")
                    print("🔍 Search: 'price of gold today', 'search for news'")
                    print("🎵 Music: 'play imagine dragons on youtube'")
                    print("💻 Apps: 'open calculator', 'close notepad'")
                    print("⏰ Reminders: 'remind me to workout|in 2 hours'")
                    print("📄 Files: 'read myfile.txt'")
                    print("💻 System: 'check system status'")
                    print("🤖 AI: 'what is artificial intelligence'")
                    continue

                if not user_input:
                    print("💡 Try asking something! Type 'help' for examples.")
                    continue

                # Process the request
                if not debug_mode:
                    print("🤔 Processing...", end="", flush=True)

                response = agent.chat(user_input, show_internal_process=debug_mode)

                if not debug_mode:
                    print(f"\r🤖 Agent: {response}              ")
                else:
                    print(f"\n🤖 Agent: {response}")

            except KeyboardInterrupt:
                print("\n\n👋 Thanks for using the Enhanced AI Agent! Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {str(e)}")
                print("💡 The agent is still running. Try another command or type 'quit' to exit.")

    except Exception as e:
        print(f"❌ Failed to initialize Enhanced AI Agent: {str(e)}")
        print("\n💡 Installation Tips:")
        print("✓ For Search: pip install duckduckgo-search")
        print("✓ For OpenAI: Set OPENAI_API_KEY environment variable")
        print("✓ For Ollama: Install Ollama and run 'ollama run llama2'")
        print("✓ Basic features work without external dependencies")


if __name__ == "__main__":
    main()

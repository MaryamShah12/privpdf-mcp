import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

from fastmcp import FastMCP
from tools import register_tools

mcp = FastMCP(name="PrivatePDF")
register_tools(mcp)

if __name__ == "__main__":
    mcp.run(transport="stdio")
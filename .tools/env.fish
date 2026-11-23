if not contains "$HOME/Desktop/dev-projects/rag-langchain-documentation-retriever/.tools" $PATH
    # Prepending path in case a system-installed binary needs to be overridden
    set -x PATH "$HOME/Desktop/dev-projects/rag-langchain-documentation-retriever/.tools" $PATH
end

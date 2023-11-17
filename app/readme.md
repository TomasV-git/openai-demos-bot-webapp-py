# Azure OpenAI Streamlit app

## deployment
```sh
az acr build --registry oaimma --resource-group rg-openai-bot --image oimma-streamlit --file WebApp.Dockerfile .
```
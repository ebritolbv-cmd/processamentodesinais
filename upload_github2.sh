#!/bin/bash

# Script para automatizar o upload dos arquivos do Desafio 1 para o GitHub

# Cores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Automação de Upload: Desafio 1 (CNN Audio Classification) ===${NC}"

# 1. Verificar se o git está inicializado
if [ ! -d ".git" ]; then
    echo -e "${BLUE}Inicializando repositório Git local...${NC}"
    git init
    # Renomear branch para main
    git branch -M main
fi

# 2. Solicitar URL do repositório se não houver remote
if ! git remote | grep -q "origin"; then
    echo -e "${GREEN}Por favor, insira a URL do seu repositório GitHub (ex: https://github.com/usuario/projeto.git):${NC}"
    read repo_url
    git remote add origin "$repo_url"
fi

# 3. Organizar arquivos
echo -e "${BLUE}Organizando arquivos para o commit...${NC}"

# Renomear README se necessário para o padrão do GitHub
if [ -f "README_Desafio1.md" ]; then
    mv README_Desafio1.md README.md
fi

# Adicionar arquivos ao stage
git add README.md desafio1_cnn.py resultado_cnn.png

# 4. Commit
echo -e "${BLUE}Criando commit...${NC}"
git commit -m "feat: add Desafio 1 - Audio Classification with CNN and MFCC"

# 5. Push
echo -e "${BLUE}Enviando para o GitHub...${NC}"
echo -e "${GREEN}Nota: Se solicitado, use seu GitHub Token como senha.${NC}"

# Tentar fazer o push
if git push -u origin main; then
    echo -e "${GREEN}=== Sucesso! Seus arquivos foram enviados para o GitHub. ===${NC}"
else
    echo -e "\033[0;31mErro ao fazer o push. Verifique suas credenciais e a URL do repositório.\033[0m"
fi

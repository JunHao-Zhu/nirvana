# Nirvana Documentation

Welcome to the Nirvana documentation! Nirvana is an LLM-powered semantic data analytics programming framework that enables natural language queries over structured and unstructured data.

## Building the Documentation

### Prerequisites

Install the documentation dependencies:

```bash
pip install -r docs/requirements.txt
```

### Build Locally

To build and serve the documentation locally:

```bash
mkdocs serve
```

The documentation will be available at `http://127.0.0.1:8000`

### Build for Production

To build static HTML files:

```bash
mkdocs build
```

The generated site will be in the `site/` directory.

### Deploy to GitHub Pages

If you want to deploy to GitHub Pages:

```bash
mkdocs gh-deploy
```

This will build the site and push it to the `gh-pages` branch.

## Documentation Structure

- **[Get Started](get_started.md)**: Quick installation and basic usage
- **[Tutorial](tutorial.md)**: Comprehensive guide to using Nirvana
- **[API Reference](api_reference.md)**: Complete API documentation
- **[Development](development.md)**: Core concepts and internals for developers

## Documentation Style

This documentation uses Material for MkDocs, similar to DSPy's documentation style. It features:

- Clean, modern design with dark mode support
- Responsive navigation with tabs and sections
- Code highlighting and copy buttons
- Search functionality
- Mobile-friendly layout

## Contributing

When contributing to the documentation:

1. Edit the markdown files in the `docs/` directory
2. Test locally with `mkdocs serve`
3. Ensure all links work and code examples are correct
4. Submit a pull request

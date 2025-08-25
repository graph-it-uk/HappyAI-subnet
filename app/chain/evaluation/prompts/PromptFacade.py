import json
from pathlib import Path

import frontmatter
from jinja2 import (Environment, FileSystemLoader, StrictUndefined,
                    TemplateError, meta)


class PromptFacade:
    
    _env = None
    
    @classmethod
    def _get_env(cls, templates_dir = "prompts/templates"):
        templates_dir = Path(__file__).parent.parent / templates_dir
        if cls._env is None:
            cls._env = Environment(
                loader=FileSystemLoader(templates_dir),
                undefined=StrictUndefined
            )
            
            
        return cls._env
            
    @staticmethod
    def get_prompt(template, **kwargs):
        env = PromptFacade._get_env()
        template_path = f"{template}.j2"
        with open(env.loader.get_source(env, template_path)[1]) as file:
            post = frontmatter.load(file)

        template = env.from_string(post.content)
        try:
            return template.render(**kwargs)
        except TemplateError as e:
            raise ValueError(f"Error rendering template: {str(e)}")
    
    
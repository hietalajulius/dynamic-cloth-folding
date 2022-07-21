import jinja2
import os

class TemplateRenderer():
    def __init__(self, directory):
        self.directory = directory
        self.loader = jinja2.FileSystemLoader(searchpath=[self.directory])
        self.template_env = jinja2.Environment(loader=self.loader)
        self.assets_dir = os.path.join(os.path.abspath("."),"env", "mujoco_templates")

    def render_template(self, template_file, **kwargs):
        template = self.template_env.get_template(template_file)
        rendered_xml = template.render(template_dir=self.assets_dir, **kwargs)
        return rendered_xml

    def render_to_file(self, template_file, target_file, **kwargs):
        xml = self.render_template(template_file, **kwargs)
        with open(target_file, "w") as f:
            f.write(xml)

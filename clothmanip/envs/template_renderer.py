import jinja2
import os

class TemplateRenderer():
    def __init__(self):
        # Set the relative template dir as base
        base_dir = os.path.dirname(os.path.abspath(__file__))
        template_dir = os.path.join(base_dir, "mujoco_templates")
        self.template_dirs = [template_dir]
        self.loader = jinja2.FileSystemLoader(searchpath=self.template_dirs)
        self.template_env = jinja2.Environment(loader=self.loader)

    def render_template(self, template_file, **kwargs):
        template = self.template_env.get_template(template_file)
        rendered_xml = template.render(template_dir=self.template_dirs[0], **kwargs)
        return rendered_xml

    def render_to_file(self, template_file, target_file, **kwargs):
        xml = self.render_template(template_file, **kwargs)
        with open(target_file, "w") as f:
            f.write(xml)

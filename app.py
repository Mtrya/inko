"""Launch the Inko demo from a source checkout or Hugging Face Space."""

# Spaces discovers the running Blocks instance in this launch module.
from demo.app import demo, launch  # noqa: F401

if __name__ == "__main__":
    launch()

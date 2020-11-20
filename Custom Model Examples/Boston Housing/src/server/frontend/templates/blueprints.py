#-----------------------------------------------------------
# Blueprint for mainPage
#-----------------------------------------------------------

from flask import Blueprint, render_template, abort
from jinja2 import TemplateNotFound

mainPage = Blueprint('apis', __name__)

#-----------------------------------------------------------
# Blueprint for apis
#-----------------------------------------------------------

apis = Blueprint('apis', __name__)

@apis.route('/<page>')
def show(page):
    try:
        return render_template('apis/%s.html' % page)
    except TemplateNotFound:
        abort(404)
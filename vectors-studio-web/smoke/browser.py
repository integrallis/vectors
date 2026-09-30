"""Exercise real Studio browser workflows against StudioFixture (no quality benchmark)."""
import argparse
import json
import re
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://127.0.0.1:8288')
parser.add_argument('--output', required=True)
args = parser.parse_args()
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
results, errors, failed_requests = [], [], []
with sync_playwright() as p:
    browser = p.chromium.launch(args=['--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
    context = browser.new_context(viewport={'width': 1440, 'height': 1000})
    context.tracing.start(screenshots=True, snapshots=True, sources=True)
    page = context.new_page()
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.on('requestfailed', lambda req: failed_requests.append({'url': req.url, 'failure': req.failure}))
    page.set_default_timeout(20000)

    def step(name, action):
        start_errors = len(errors)
        try:
            action()
            assert len(errors) == start_errors, errors[start_errors:]
            results.append({'name': name, 'status': 'passed'})
        except Exception as error:
            results.append({'name': name, 'status': 'failed', 'error': str(error)})
        finally:
            page.screenshot(path=str(out / (name + '.png')), full_page=True)

    def collections():
        page.goto(args.url + '/collections')
        expect(page.locator('#row-docs')).to_contain_text('96 docs')
        expect(page.locator('#row-empty')).to_contain_text('0 docs')
        page.locator('#row-docs a').click()
        expect(page.locator('.doc-list tbody tr')).to_have_count(25)
        expect(page.locator('.doc-list-header')).to_contain_text('Showing 1–25 of 96')
        page.get_by_role('button', name='Next ›', exact=True).click()
        expect(page.locator('.doc-list-header')).to_contain_text('Showing 26–50 of 96')
        page.get_by_role('button', name='10', exact=True).click()
        expect(page.locator('.doc-list tbody tr')).to_have_count(10)
        expect(page.locator('.pager-info')).to_have_text('Page 1 of 10')
    step('collections-pagination', collections)

    def inspect():
        page.goto(args.url + '/collections/docs/documents/doc-0')
        expect(page.locator('h1')).to_have_text('doc-0')
        expect(page.locator('.viewer')).to_contain_text('Literal <script>window.fixtureInjection=true</script> & text')
        expect(page.locator('main')).to_contain_text('group-0')
        assert page.evaluate('window.fixtureInjection === undefined')
    step('document-inspection', inspect)

    def empty():
        page.goto(args.url + '/collections/empty')
        expect(page.locator('#preview-pane')).to_contain_text('No documents on this page.')
        expect(page.get_by_role('button', name='Next ›', exact=True)).to_be_disabled()
    step('empty-collection', empty)

    def projection(algo):
        if algo == 'pca':
            page.goto(args.url + '/collections/docs/projector')
            expect(page.locator('.projector-canvas canvas')).to_be_visible()
        else:
            with page.expect_response(lambda r: r.url.endswith('/api/projections') and r.request.method == 'POST') as response:
                page.locator('[data-algo="' + algo + '"]').click()
            assert response.value.status == 202, response.value.text()
        expect(page.locator('.projector-status')).to_have_text(re.compile(r'^done ·'), timeout=90000)
        expect(page.locator('#data-sphereize')).to_be_enabled()
    for algo in ['pca', 'tsne', 'umap']:
        step('projection-' + algo, lambda algo=algo: projection(algo))

    def inspector():
        expect(page.locator('#ins-mmr-lambda-field')).to_be_hidden()
        page.locator('#ins-query').fill('doc-0')
        page.locator('#ins-search').click()
        expect(page.locator('#ins-hits li')).to_have_count(10)
        expect(page.locator('#ins-isolate')).to_be_enabled()
        page.locator('#ins-isolate').click()
        page.locator('#ins-show-all').click()
        page.locator('#ins-mmr').check()
        expect(page.locator('#ins-mmr-lambda-field')).to_be_visible()
        expect(page.locator('#ins-hits li')).to_have_count(10)
        page.locator('#ins-clear').click()
        expect(page.locator('#ins-hits li')).to_have_count(0)
    step('projector-neighbors', inspector)

    def two_d():
        with page.expect_response(lambda r: r.url.endswith('/api/projections') and r.request.method == 'POST'):
            page.get_by_role('radio', name='2D', exact=True).click()
        expect(page.locator('.projector-status')).to_have_text(re.compile(r'^done ·'), timeout=90000)
        assert page.locator('[data-dim="2"]').get_attribute('class').find('is-active') >= 0
    step('projection-2d', two_d)

    def datasets():
        page.goto(args.url + '/datasets')
        expect(page.locator('h1')).to_contain_text('Sample datasets')
        expect(page.locator('main')).to_contain_text('DBpedia')
    step('dataset-browser', datasets)

    def providers():
        page.goto(args.url + '/providers')
        expect(page.locator('main')).to_contain_text('provider')
    step('provider-browser', providers)

    def delete_empty():
        page.goto(args.url + '/collections')
        page.once('dialog', lambda dialog: dialog.accept())
        page.locator('#row-empty').get_by_role('button', name='Delete').click()
        expect(page.locator('#row-empty')).to_have_count(0)
        page.reload()
        expect(page.locator('#row-empty')).to_have_count(0)
    step('delete-collection', delete_empty)
    context.tracing.stop(path=str(out / 'trace.zip'))
    report = {'browser': browser.version, 'results': results, 'page_errors': errors, 'failed_requests': failed_requests}
    (out / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    browser.close()
print(json.dumps(results, indent=2))
raise SystemExit(1 if any(r['status'] == 'failed' for r in results) else 0)

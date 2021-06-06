# config

params = {
        # crypto setup
        'currency_1': 'BTC', # bitcoin
        'ref_currency': 'USD',
        'ma': 25,
        # api setup       #120 secs
        'api_call_period': 10, # in secs!
	'website_length': 100,
}

config = {
        # api auth
        'api_key': ':)',
        'api_secret': ':)',
        'api_pass': ':)',
        # kafka
        'kafka_broker': 'localhost:9092',
        # topics
        'topic_1': 'topic_{0}'.format(params['currency_1']),
        'topic_2': 'topic_{0}_ma_{1}'.format(params['currency_1'], params['ma']),
}


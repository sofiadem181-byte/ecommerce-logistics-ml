#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Εισαγωγή της βιβλιοθήκης pandas για ανάλυση και διαχείριση δεδομένων
import pandas as pd

# Φόρτωση του dataset παραγγελιών (orders),το οποίο περιλαμβάνει πληροφορίες
# σχετικά με την κατάσταση και τα χρονικά σημεία εξέλιξης κάθε παραγγελίας
orders = pd.read_csv("olist_orders_dataset.csv")

# Εμφάνιση των πρώτων εγγραφών του συνόλου δεδομένων για προκαταρκτική διερεύνηση
orders.head()


# In[2]:


# Έλεγχος της δομής του dataset παραγγελιών,
# Παρουσίαση τύπων δεδομένων και πλήθους μη κενών τιμών για κάθε μεταβλητή
orders.info()


# In[3]:


# Περιγραφική σύνοψη των κατηγορικών (μη αριθμητικών) μεταβλητών του dataset παραγγελιών
# Παρουσιάζονται το πλήθος εγγραφών, οι μοναδικές τιμές και η συχνότερη εμφάνιση κάθε μεταβλητής
orders.describe()


# In[4]:


# Φόρτωση του dataset πελατών, το οποίο περιλαμβάνει δημογραφικές και γεωγραφικές πληροφορίες
customers = pd.read_csv("olist_customers_dataset.csv")

# Εμφάνιση των πρώτων εγγραφών για αρχική διερεύνηση των δεδομένων
customers.head()


# In[5]:


# Έλεγχος της δομής του dataset πελατών
# Παρουσίαση τύπων δεδομένων και ύπαρξης κενών τιμών στις μεταβλητές
customers.info()


# In[6]:


# Φόρτωση του dataset προϊόντων ανά παραγγελία (τιμές προϊόντων και κόστος μεταφοράς)
items = pd.read_csv("olist_order_items_dataset.csv")
# Αρχική επισκόπηση των δεδομένων
items.head()
# Έλεγχος δομής, τύπων δεδομένων και πληρότητας των μεταβλητών
items.info()


# In[7]:


# Φόρτωση του dataset προϊόντων, το οποίο περιλαμβάνει πληροφορίες
# σχετικά με την κατηγορία και τα βασικά χαρακτηριστικά κάθε προϊόντος
products = pd.read_csv("olist_products_dataset.csv")

products.head()


# In[8]:


# Ενοποίηση δεδομένων παραγγελιών και πελατών
# Συγχώνευση μέσω μοναδικού αναγνωριστικού customer_id
# Επιλέγεται εσωτερική σύζευξη (inner join) ώστε να διατηρηθούν
# μόνο οι παραγγελίες που αντιστοιχούν σε έγκυρους πελάτες
orders_customers = pd.merge(
    orders,
    customers,
    on="customer_id",
    how="inner"
)


# In[9]:


# Ενοποίηση δεδομένων παραγγελιών, πελατών και προϊόντων με βάση το order_id
# Δημιουργία του τελικού συνόλου δεδομένων για ανάλυση
full_data = pd.merge(
    orders_customers,
    items,
    on="order_id",
    how="inner"
)

# Προβολή των πρώτων γραμμών του dataset
full_data.head()


# In[10]:


# Έλεγχος αριθμού ελλιπών τιμών ανά μεταβλητή
full_data.isnull().sum()


# In[11]:


# Υπολογισμός ποσοστού ελλιπών τιμών ανά μεταβλητή
(full_data.isnull().sum() / len(full_data)) * 100


# In[12]:


# Υπολογισμός μοναδικού πλήθους παραγγελιών στο dataset
full_data["order_id"].nunique()


# In[13]:


# Κατανομή καταστάσεων παραγγελιών
full_data["order_status"].value_counts()


# In[14]:


# Ποσοστιαία κατανομή καταστάσεων παραγγελιών
full_data["order_status"].value_counts(normalize=True) * 100


# In[15]:


# Οι 10 πολιτείες με το μεγαλύτερο πλήθος παραγγελιών
full_data["customer_state"].value_counts().head(10)


# In[16]:


# Υπολογισμός συνολικής αξίας παραγγελίας ως άθροισμα
# της τιμής προϊόντος και του κόστους μεταφοράς
full_data["order_value"] = full_data["price"] + full_data["freight_value"]


# In[17]:


# Βασική στατιστική σύνοψη της αξίας παραγγελίας
full_data["order_value"].describe()


# In[18]:


# Δημιουργία δυαδικής μεταβλητής για την κατάσταση παράδοσης
# 1: παραγγελία παραδόθηκε, 0: διαφορετική κατάσταση
full_data["delivered_binary"] = full_data["order_status"].apply(
    lambda x: 1 if x == "delivered" else 0
)

# Κατανομή παραγγελιών ως προς την κατάσταση παράδοσης
full_data["delivered_binary"].value_counts()


# In[19]:


# Μετατροπή των χρονικών μεταβλητών σε τύπο datetime
full_data["order_delivered_customer_date"] = pd.to_datetime(
    full_data["order_delivered_customer_date"]
)
full_data["order_estimated_delivery_date"] = pd.to_datetime(
    full_data["order_estimated_delivery_date"]
)

# Υπολογισμός ημερών καθυστέρησης παράδοσης ως η διαφορά
# μεταξύ πραγματικής και εκτιμώμενης ημερομηνίας παράδοσης
full_data["delivery_delay_days"] = (
    full_data["order_delivered_customer_date"]
    - full_data["order_estimated_delivery_date"]
).dt.days


# In[20]:


# Επιλογή βασικών μεταβλητών για περαιτέρω ανάλυση και μοντελοποίηση
analysis_data = full_data[
    [
        "customer_state",
        "order_value",
        "delivered_binary",
        "delivery_delay_days"
    ]
]
analysis_data.head()


# In[21]:


# Εισαγωγή της βιβλιοθήκης matplotlib για οπτικοποίηση δεδομένων
import matplotlib.pyplot as plt

# Οπτικοποίηση της κατανομής των παραγγελιών ως προς την κατάσταση ολοκλήρωσης
status_counts = full_data["order_status"].value_counts()

status_counts.plot(kind="bar")
plt.title("Κατανομή κατάστασης παραγγελιών")
plt.xlabel("Κατάσταση παραγγελίας")
plt.ylabel("Πλήθος παραγγελιών")
plt.show()


# In[22]:


# Οπτικοποίηση των 10 πολιτειών με το μεγαλύτερο πλήθος παραγγελιών
state_counts = full_data["customer_state"].value_counts().head(10)

state_counts.plot(kind="bar")
plt.title("Παραγγελίες ανά πολιτεία (Top 10)")
plt.xlabel("Πολιτεία")
plt.ylabel("Πλήθος παραγγελιών")
plt.show()


# In[23]:


# Ιστόγραμμα κατανομής αξίας παραγγελίας
plt.hist(full_data["order_value"], bins=30)
plt.title("Κατανομή αξίας παραγγελίας")
plt.xlabel("Αξία παραγγελίας")
plt.ylabel("Συχνότητα")
plt.show()


# In[24]:


analysis_data = full_data[
    ["order_value", "delivery_delay_days", "delivered_binary"]
].dropna()

# Επιλογή χαρακτηριστικών (features) και μεταβλητής στόχου
X = analysis_data[["order_value", "delivery_delay_days"]]
y = analysis_data["delivered_binary"]

# Διαχωρισμός δεδομένων σε σύνολα εκπαίδευσης και ελέγχου
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Εκπαίδευση μοντέλου λογιστικής παλινδρόμησης
model = LogisticRegression()
model.fit(X_train, y_train)


# In[25]:


# Αξιολόγηση απόδοσης μοντέλου
from sklearn.metrics import accuracy_score, classification_report
y_pred = model.predict(X_test)
# Αναλυτική αξιολόγηση της απόδοσης του μοντέλου ανά κατηγορία
print(classification_report(y_test, y_pred))


# In[26]:


# Υπολογισμός της μήτρας σύγχυσης (confusion matrix) για το μοντέλο ταξινόμησης
# Το αποτέλεσμα δείχνει ότι το μοντέλο ταξινομεί όλες τις παρατηρήσεις
# ως "delivered" (κλάση 1), χωρίς να προβλέπει καμία μη παραδομένη παραγγελία (κλάση 0),
# λόγω της έντονης ανισορροπίας μεταξύ των κλάσεων στο σύνολο δεδομένων
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


# In[28]:


# Αναλυτική αξιολόγηση της απόδοσης του μοντέλου ανά κατηγορία
# Παρατηρείται υψηλή ακρίβεια για την πλειοψηφική κατηγορία (delivered),
# ενώ το μοντέλο αποτυγχάνει πλήρως να αναγνωρίσει μη παραδομένες παραγγελίες,
# γεγονός που επιβεβαιώνει το πρόβλημα της ανισορροπίας κλάσεων (class imbalance)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# In[29]:


# Έλεγχος ενδεικτικών προβλέψεων του μοντέλου
# Διαπιστώνεται ότι όλες οι προβλέψεις ανήκουν στην κλάση "delivered" (1),
# επιβεβαιώνοντας τη μονομερή συμπεριφορά του μοντέλου
y_pred[:10]


# In[30]:


# Μήτρα σύγχυσης του αρχικού (μη εξισορροπημένου) μοντέλου
# Το μοντέλο ταξινομεί όλες τις παρατηρήσεις στην πλειοψηφική κατηγορία ("delivered"),
# χωρίς σωστές προβλέψεις για μη παραδομένες παραγγελίες
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


# In[31]:


# Υπολογισμός συνολικής ακρίβειας του αρχικού μοντέλου
# Η υψηλή τιμή της ακρίβειας είναι παραπλανητική,
# καθώς οφείλεται στην επικράτηση της πλειοψηφικής κλάσης
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[32]:


# Εκπαίδευση λογιστικής παλινδρόμησης με εξισορρόπηση κλάσεων (class_weight='balanced')
# Στόχος είναι η βελτίωση της πρόβλεψης της μειοψηφικής κατηγορίας (μη παραδομένες παραγγελίες),
# ακόμη και αν αυτό οδηγήσει σε μείωση της συνολικής ακρίβειας
from sklearn.linear_model import LogisticRegression

model_balanced = LogisticRegression(class_weight='balanced')
model_balanced.fit(X_train, y_train)

y_pred_balanced = model_balanced.predict(X_test)


# In[33]:


# Αξιολόγηση του εξισορροπημένου μοντέλου ταξινόμησης
# Η χρήση εξισορρόπησης κλάσεων οδηγεί σε καλύτερη ανίχνευση μη παραδομένων παραγγελιών,
# ενώ η συνολική ακρίβεια μειώνεται, λόγω της αυξημένης έμφασης στη μειοψηφική κατηγορία
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_balanced))


# In[34]:


# Επιλογή των απαραίτητων μεταβλητών για ανάλυση συσχετίσεων προϊόντων (Market Basket Analysis)
# Κάθε παραγγελία αντιστοιχεί σε ένα σύνολο προϊόντων
transactions = items[['order_id', 'product_id']]


# In[35]:


# Ομαδοποίηση προϊόντων ανά παραγγελία
# Δημιουργία "καλαθιού αγορών" για κάθε παραγγελία
basket = transactions.groupby('order_id')['product_id'].apply(list)


# In[36]:


# Μετασχηματισμός των συναλλαγών σε δυαδικό πίνακα παρουσίας/απουσίας προϊόντων,
# προκειμένου να καταστεί δυνατή η εφαρμογή αλγορίθμων εξόρυξης συσχετίσεων,
# όπως ο αλγόριθμος Apriori
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_array = te.fit(basket).transform(basket)

basket_encoded = pd.DataFrame(te_array, columns=te.columns_)


# In[37]:


# Εγκατάσταση και εισαγωγή της βιβλιοθήκης mlxtend,
# η οποία χρησιμοποιείται για εξόρυξη συχνών συνόλων και κανόνων συσχέτισης
get_ipython().system('conda install -c conda-forge mlxtend -y')


# In[38]:


from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


# In[39]:


# Κωδικοποίηση των καλαθιών αγορών σε δυαδική μορφή (True/False),
# όπου κάθε στήλη αντιστοιχεί σε προϊόν και κάθε γραμμή σε μία παραγγελία
te = TransactionEncoder()
te_array = te.fit(basket).transform(basket)

basket_encoded = pd.DataFrame(te_array, columns=te.columns_)
basket_encoded.head()


# In[40]:


# Έλεγχος του πλήθους προϊόντων ανά παραγγελία
# Παρατηρείται ότι πολλές παραγγελίες περιλαμβάνουν ένα μόνο προϊόν,
# γεγονός σύνηθες στο συγκεκριμένο σύνολο δεδομένων
# Ο έλεγχος αυτός επιβεβαιώνει τη δομή των συναλλαγών
# πριν την εφαρμογή αλγορίθμων εξόρυξης συσχετίσεων (Apriori)
basket_encoded.sum(axis=1).head(10)


# In[41]:


# Προβολή μόνο των προϊόντων που εμφανίζονται στις πρώτες παραγγελίες,
# για επιβεβαίωση της ορθής κωδικοποίησης των δεδομένων
basket_encoded.head().loc[:, basket_encoded.head().any()]


# In[42]:


# Εμφάνιση των συχνότερων καλαθιών αγορών (ίδιων συνδυασμών προϊόντων)
# Κάθε εγγραφή αντιστοιχεί σε ένα μοναδικό σύνολο προϊόντων ανά παραγγελία
basket.value_counts().head(10)


# In[43]:


# Κατανομή του πλήθους προϊόντων ανά καλάθι αγορών
# Παρουσιάζεται πόσες παραγγελίες περιέχουν 1, 2, 3 κ.ο.κ. προϊόντα
basket.apply(len).value_counts().sort_index().head(10)


# In[44]:


# Φιλτράρισμα καλαθιών που περιλαμβάνουν τουλάχιστον δύο διαφορετικά προϊόντα
# Το βήμα αυτό είναι απαραίτητο για την εξόρυξη κανόνων συσχέτισης,
# καθώς τα καλάθια με ένα μόνο προϊόν δεν παράγουν συσχετίσεις
basket_2plus = basket[basket.apply(lambda x: len(set(x)) >= 2)]

print("Σύνολο καλαθιών:", len(basket))
print("Καλάθια με ≥2 προϊόντα:", len(basket_2plus))


# In[45]:


from mlxtend.preprocessing import TransactionEncoder
import pandas as pd


# In[46]:


# Επανακωδικοποίηση των φιλτραρισμένων καλαθιών (≥2 προϊόντα) σε δυαδική μορφή,
# ώστε να χρησιμοποιηθούν ως είσοδος στον αλγόριθμο Apriori
te = TransactionEncoder()
te_array = te.fit(basket_2plus).transform(basket_2plus)

basket_encoded = pd.DataFrame(te_array, columns=te.columns_)


# In[47]:


# Προβολή του δυαδικά κωδικοποιημένου πίνακα συναλλαγών
# μετά το φιλτράρισμα καλαθιών με τουλάχιστον δύο προϊόντα
basket_encoded.head()


# In[48]:


# Εξόρυξη συχνών συνόλων προϊόντων με τον αλγόριθμο Apriori,
# εφαρμόζοντας κατώφλι υποστήριξης ίσο με 0.5%
from mlxtend.frequent_patterns import apriori, association_rules

# Συχνά σύνολα προϊόντων (frequent itemsets)
frequent_itemsets = apriori(
    basket_encoded,
    min_support=0.005,      # κατώφλι 0.005 (0.5%)
    use_colnames=True
)

frequent_itemsets = frequent_itemsets.sort_values("support", ascending=False)

frequent_itemsets.head(10)


# In[49]:


# Δημιουργία κανόνων συσχέτισης από τα συχνά σύνολα προϊόντων
# με χρήση του δείκτη lift, ώστε να εντοπιστούν μη τυχαίες συσχετίσεις
rules = association_rules(
    frequent_itemsets,
    metric="lift",
    min_threshold=1.0
)

# Τα αποτελέσματα ταξινομούνται βάσει lift και confidence
# για ευκολότερη ερμηνεία των ισχυρότερων κανόνων
rules = rules.sort_values(["lift", "confidence"], ascending=False)
rules.head(10)


# In[50]:


# Δημιουργία απλοποιημένου πίνακα κανόνων συσχέτισης
# και μετατροπή των συνόλων προϊόντων σε λίστες,
# ώστε να διευκολυνθεί η ερμηνεία και παρουσίαση των αποτελεσμάτων
rules_small = rules[["antecedents","consequents","support","confidence","lift"]].copy()

rules_small["antecedents"] = rules_small["antecedents"].apply(lambda x: list(x))
rules_small["consequents"] = rules_small["consequents"].apply(lambda x: list(x))

rules_small.head(10)


# In[51]:


# Φιλτράρισμα των κανόνων συσχέτισης βάσει κατωφλίων confidence και lift,
# με στόχο τη διατήρηση μόνο των πιο αξιόπιστων και ουσιαστικών συσχετίσεων
rules_filtered = rules[
    (rules["confidence"] >= 0.20) &   # π.χ. τουλάχιστον 20%
    (rules["lift"] >= 1.20)           # π.χ. lift αρκετά πάνω από 1
].sort_values(["lift","confidence"], ascending=False)

rules_filtered.head(10)


# In[ ]:





# In[ ]:




